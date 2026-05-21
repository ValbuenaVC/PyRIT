# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario class for grouping and executing multiple AtomicAttacks.

This module provides the Scenario class that orchestrates the execution of multiple
AtomicAttack instances sequentially, enabling comprehensive security testing campaigns.
"""

import copy
import json
import logging
import textwrap
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union, cast, get_origin

from tqdm.auto import tqdm

from pyrit.common import REQUIRED_VALUE, Parameter, apply_defaults
from pyrit.common.deprecation import print_deprecation_message
from pyrit.common.parameter import coerce_value, validate_param_type
from pyrit.common.utils import to_sha256
from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.memory import CentralMemory
from pyrit.memory.memory_models import ScenarioResultEntry
from pyrit.models import AttackOutcome, AttackResult, SeedAttackGroup
from pyrit.models.scenario_result import ScenarioIdentifier, ScenarioResult
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.common.target_requirements import TargetRequirements
from pyrit.registry import ScorerRegistry
from pyrit.scenario.core.atomic_attack import AtomicAttack
from pyrit.scenario.core.attack_technique import AttackTechnique
from pyrit.scenario.core.dataset_configuration import DatasetConfiguration
from pyrit.scenario.core.input_schema import RoleDescriptor
from pyrit.scenario.core.scenario_step import ScenarioStep, ScenarioStepResult
from pyrit.scenario.core.scenario_strategy import ScenarioStrategy
from pyrit.scenario.core.scenario_target_defaults import get_default_scorer_target
from pyrit.scenario.core.strategy_graph import (
    PolicyAction,
    StrategyGraph,
    StrategyPolicy,
)
from pyrit.score import (
    Scorer,
    SelfAskRefusalScorer,
    SelfAskTrueFalseScorer,
    TrueFalseCompositeScorer,
    TrueFalseInverterScorer,
    TrueFalseScoreAggregator,
    TrueFalseScorer,
)

if TYPE_CHECKING:
    from pyrit.identifiers import ComponentIdentifier
    from pyrit.scenario.core.attack_technique_factory import AttackTechniqueFactory

logger = logging.getLogger(__name__)


class BaselineAttackPolicy(Enum):
    """
    Declares how a scenario type treats the default baseline atomic attack.

    The baseline is a plain ``PromptSendingAttack`` that sends each objective unmodified,
    used as a comparison point against the scenario's strategies. Each scenario class
    declares its policy via ``Scenario.BASELINE_ATTACK_POLICY``; callers can still override
    at runtime via ``initialize_async(include_baseline=...)`` for the ``Enabled`` and
    ``Disabled`` states.
    """

    #: Supported and prepended automatically. Caller can opt out at runtime.
    Enabled = "enabled"

    #: Supported but only included when the caller explicitly requests it.
    Disabled = "disabled"

    #: Not supported. Explicit ``include_baseline=True`` at runtime raises ``ValueError``.
    Forbidden = "forbidden"


def _assert_json_serializable(*, params: dict[str, Any]) -> None:
    """
    Raise if any value in ``params`` cannot round-trip through JSON.

    Stage 5 stores ``params`` on ``ScenarioIdentifier.init_data`` for resume
    validation; the underlying memory column is JSON. Catching unserializable
    values here gives a clear error rather than a database failure.

    Args:
        params (dict[str, Any]): Effective parameters to validate.

    Raises:
        ValueError: If any value is not JSON-serializable.
    """
    try:
        json.dumps(params)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Scenario params contain a non-JSON-serializable value (cannot persist for resume): {exc}. "
            f"Use only JSON-safe types (str, int, float, bool, list, dict, None) for scenario parameters."
        ) from exc


def _format_param_key_diff(*, stored: dict[str, Any], current: dict[str, Any]) -> str:
    """
    Render the set-level difference between two param dicts as a short string.

    Lists only key names (no values) so secrets or large blobs in scenario
    parameters do not leak into logs.

    Args:
        stored (dict[str, Any]): Persisted params from the previous run.
        current (dict[str, Any]): Effective params for the current run.

    Returns:
        str: A short summary like ``"added: x, y; removed: z; changed: max_turns"``.
    """
    parts: list[str] = []
    added = sorted(set(current) - set(stored))
    removed = sorted(set(stored) - set(current))
    changed = sorted(k for k in set(stored) & set(current) if stored[k] != current[k])
    if added:
        parts.append(f"added: {', '.join(added)}")
    if removed:
        parts.append(f"removed: {', '.join(removed)}")
    if changed:
        parts.append(f"changed: {', '.join(changed)}")
    return "; ".join(parts) if parts else "no diff details"


class Scenario(ABC):
    """
    Groups and executes multiple AtomicAttack instances sequentially.

    A Scenario represents a comprehensive testing campaign composed of multiple
    atomic attack tests (AtomicAttacks). It executes each AtomicAttack in sequence and
    aggregates the results into a ScenarioResult.
    """

    #: Capability requirements placed on ``objective_target``. Subclasses override to declare
    #: what the scenario needs. Validated in ``initialize_async`` once the target is supplied.
    TARGET_REQUIREMENTS: ClassVar[TargetRequirements] = TargetRequirements()

    #: How this scenario type treats the default baseline atomic attack. Subclasses override
    #: when their semantics call for a different default (``Disabled``) or when a baseline
    #: is meaningless for the comparison the scenario performs (``Forbidden``). Resolved in
    #: ``initialize_async`` and overridable per run via ``include_baseline`` for the
    #: ``Enabled`` and ``Disabled`` states; ``Forbidden`` is a hard constraint and a
    #: caller-supplied ``include_baseline=True`` raises ``ValueError``.
    BASELINE_ATTACK_POLICY: ClassVar[BaselineAttackPolicy] = BaselineAttackPolicy.Enabled

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Warn once per subclass that still overrides the legacy step builder.

        ``_get_atomic_attacks_async`` was renamed to :meth:`_get_steps_async`
        in PyRIT 0.15; the old name keeps working through a passthrough but
        will be removed in 0.16. We detect the override at class-creation
        time so authors see the deprecation as soon as their subclass module
        is imported instead of only on the next ``run_async`` call.

        Args:
            **kwargs (Any): Forwarded to ``ABC.__init_subclass__``.
        """
        super().__init_subclass__(**kwargs)
        overrides_legacy = "_get_atomic_attacks_async" in cls.__dict__
        overrides_new = "_get_steps_async" in cls.__dict__
        if overrides_legacy and not overrides_new:
            print_deprecation_message(
                old_item=f"{cls.__module__}.{cls.__qualname__}._get_atomic_attacks_async",
                new_item=f"{cls.__module__}.{cls.__qualname__}._get_steps_async",
                removed_in="0.16.0",
            )

    @classmethod
    def _get_additional_scoring_questions(cls) -> Sequence[Path]:
        """
        Paths to additional true/false question prompts for objective scoring.

        These prompts are used in the default scenario scorer in addition to a simple self-ask scorer.

        Returns:
            Sequence[Path]: Paths to true/false question prompts, or an empty sequence to use the default scorer.
        """
        return []

    def __init__(
        self,
        *,
        name: str = "",
        version: int,
        strategy_class: type[ScenarioStrategy],
        objective_scorer: Scorer,
        scenario_result_id: Optional[Union[uuid.UUID, str]] = None,
        include_default_baseline: bool | None = None,  # Deprecated. Will be removed in 0.16.0.
    ) -> None:
        """
        Initialize a scenario.

        Args:
            name (str): Descriptive name for the scenario.
            version (int): Version number of the scenario.
            strategy_class (Type[ScenarioStrategy]): The strategy enum class for this scenario.
            objective_scorer (Scorer): The objective scorer used to evaluate attack results.
            scenario_result_id (Optional[Union[uuid.UUID, str]]): Optional ID of an existing scenario result to resume.
                Can be either a UUID object or a string representation of a UUID.
                If provided and found in memory, the scenario will resume from prior progress.
                All other parameters must still match the stored scenario configuration.
            include_default_baseline (bool | None): **Deprecated.** Will be removed in 0.16.0.
                Pass ``include_baseline`` to ``initialize_async`` instead. When set, the value is
                used as the effective ``include_baseline`` for the next ``initialize_async`` call
                unless that call passes its own ``include_baseline``.

        Note:
            Attack runs are populated by calling initialize_async(), which invokes the
            subclass's _get_steps_async() method (or, for legacy subclasses still
            overriding the deprecated _get_atomic_attacks_async, the base
            _get_steps_async delegates to that legacy override).

            The scenario description is automatically extracted from the class's docstring (__doc__)
            with whitespace normalized for display.
        """
        from pyrit.registry.base import ClassRegistryEntry

        description = ClassRegistryEntry.description_from_docstring(self.__class__)

        self._identifier = ScenarioIdentifier(
            name=type(self).__name__, scenario_version=version, description=description
        )

        # Store strategy configuration for use in initialize_async
        self._strategy_class = strategy_class

        # These will be set in initialize_async
        self._objective_target: Optional[PromptTarget] = None
        self._objective_target_identifier: Optional[ComponentIdentifier] = None
        self._memory_labels: dict[str, str] = {}
        self._max_concurrency: int = 1
        self._max_retries: int = 0

        self._objective_scorer = objective_scorer
        self._objective_scorer_identifier = objective_scorer.get_identifier()

        self._name = name if name else type(self).__name__
        self._memory = CentralMemory.get_memory_instance()
        self._atomic_attacks: list[AtomicAttack] = []
        self._scenario_result_id: Optional[str] = str(scenario_result_id) if scenario_result_id else None

        # Store prepared strategies for use in _get_steps_async
        self._scenario_strategies: list[ScenarioStrategy] = []

        # Maps atomic_attack_name → display_group for user-facing aggregation
        self._display_group_map: dict[str, str] = {}

        # Custom parameters: declared via supported_parameters(), populated via set_params_from_args().
        self.params: dict[str, Any] = {}
        self._declarations_validated: bool = False

        # Resolved effective baseline inclusion for the current run. Set in initialize_async
        # before _get_steps_async is awaited so overrides can read it.
        self._include_baseline: bool = False

        # Phase 5: state-machine view over the scenario's steps. Built lazily in
        # _execute_scenario_async from self._atomic_attacks after the resume filter
        # has been applied. Stays None until the first execution attempt.
        # The default ``_build_execution_graph`` uses ``int`` as the state type; we
        # store as ``StrategyGraph[ScenarioStep, Any]`` so subclasses that override
        # the builder with a string/Enum state type can stash their graph here too
        # without invariance fights.
        self._execution_graph: Optional[StrategyGraph[ScenarioStep, Any]] = None

        # Deprecated constructor-time baseline override. Will be removed in 0.16.0, along
        # with the include_default_baseline kwarg above and the legacy fallback branch in
        # initialize_async. Subclass shims set this attribute directly to avoid double-warning.
        self._legacy_include_baseline: bool | None = None
        if include_default_baseline is not None:
            print_deprecation_message(
                old_item="Scenario(include_default_baseline=...)",
                new_item="Scenario.initialize_async(include_baseline=...)",
                removed_in="0.16.0",
            )
            self._legacy_include_baseline = include_default_baseline

    @property
    def name(self) -> str:
        """Get the name of the scenario."""
        return self._name

    @property
    def atomic_attack_count(self) -> int:
        """Get the number of atomic attacks in this scenario."""
        return len(self._atomic_attacks)

    @property
    def execution_graph(self) -> Optional[StrategyGraph[ScenarioStep, Any]]:
        """
        The ``StrategyGraph`` driving this scenario's current execution attempt.

        Built in ``_execute_scenario_async`` from the steps that remain after the
        resume filter; ``None`` before the first call to ``run_async`` (or any
        time outside of an active execution attempt).

        Subclasses can override ``_build_execution_graph`` to declare a richer
        state-machine policy; the default uses ``_build_default_linear_policy``
        to wrap ``self._atomic_attacks`` in a linear traversal that matches the
        legacy flat ``_execute_scenario_async`` loop exactly.
        """
        return self._execution_graph

    @property
    def execution_history(self) -> list[ScenarioStepResult]:
        """
        Ordered list of step results produced by the current execution attempt.

        Empty when no graph has been built yet, or between retry attempts that
        reset the graph. Each entry is the ``ScenarioStepResult`` yielded by a
        step's policy action, in execution order.
        """
        if self._execution_graph is None:
            return []
        return [result for _, result in self._execution_graph.history]

    @classmethod
    @abstractmethod
    def get_strategy_class(cls) -> type[ScenarioStrategy]:
        """
        Get the strategy enum class for this scenario.

        This abstract method must be implemented by all scenario subclasses to return
        the ScenarioStrategy enum class that defines the available attack strategies
        for the scenario.

        Returns:
            Type[ScenarioStrategy]: The strategy enum class (e.g., FoundryStrategy, EncodingStrategy).
        """

    @classmethod
    @abstractmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        """
        Get the default strategy used when no strategies are specified.

        This abstract method must be implemented by all scenario subclasses to return
        the default aggregate strategy (like EASY, ALL) used when scenario_strategies
        parameter is None.

        Returns:
            ScenarioStrategy: The default aggregate strategy (e.g., FoundryStrategy.EASY, EncodingStrategy.ALL).
        """

    @classmethod
    @abstractmethod
    def default_dataset_config(cls) -> DatasetConfiguration:
        """
        Return the default dataset configuration for this scenario.

        This abstract method must be implemented by all scenario subclasses to return
        a DatasetConfiguration specifying the default datasets to use when no
        dataset_config is provided by the user.

        Returns:
            DatasetConfiguration: The default dataset configuration.
        """

    @classmethod
    def supported_parameters(cls) -> list[Parameter]:
        """
        Override to declare custom parameters this scenario accepts.

        Declared parameters flow from CLI/config through ``set_params_from_args``
        into ``self.params`` before ``initialize_async()`` runs. Implemented as
        a classmethod so ``--list-scenarios`` can introspect without instantiating.

        Note: ``PyRITInitializer.supported_parameters`` is an instance ``@property``;
        this asymmetry is intentional pending a future alignment.

        Returns:
            list[Parameter]: Declared parameters (default: empty list).
        """
        return []

    @classmethod
    def input_schema(cls) -> list[RoleDescriptor]:
        """
        Override to declare rich-object ``__init__`` inputs the wizard should elicit.

        Returns a ``list[RoleDescriptor]`` describing arguments that
        :meth:`__init__` accepts beyond the standard scenario plumbing
        (``scenario_result_id``, ``params``, ``memory_labels``). Each descriptor
        carries a :class:`RoleTag` declaring how the role is elicited
        (scalar, choice, registry reference, factory spec, or opaque instance).

        This is intentionally orthogonal to :meth:`supported_parameters`:

        * :meth:`supported_parameters` declares **scalar** arguments to
          :meth:`initialize_async` (CLI ``--kebab-flag`` surface, unchanged).
        * :meth:`input_schema` declares **rich-object** arguments to
          :meth:`__init__` (wizard / programmatic surface).

        Default returns ``[]``; most scenarios accept no rich-object inputs
        beyond the standard plumbing.

        Returns:
            list[RoleDescriptor]: Declared roles (default: empty list).
        """
        return []

    def _get_attack_technique_factories(self) -> dict[str, "AttackTechniqueFactory"]:
        """
        Return the attack technique factories for this scenario.

        Each key is a technique name (matching a strategy enum value) and each
        value is an ``AttackTechniqueFactory`` that can produce an
        ``AttackTechnique`` for that technique.

        The base implementation lazily populates the
        ``AttackTechniqueRegistry`` singleton with core techniques (via
        ``ScenarioTechniqueRegistrar``) and returns all registered factories.
        Subclasses may override to add, remove, or replace factories.

        Returns:
            dict[str, AttackTechniqueFactory]: Mapping of technique name to factory.
        """
        from pyrit.scenario.core.scenario_techniques import register_scenario_techniques

        register_scenario_techniques()

        from pyrit.registry.object_registries.attack_technique_registry import AttackTechniqueRegistry

        return AttackTechniqueRegistry.get_registry_singleton().get_factories()

    def _build_display_group(self, *, technique_name: str, seed_group_name: str) -> str:
        """
        Build the display-group label for an atomic attack.

        Each ``AtomicAttack`` has a unique ``atomic_attack_name`` (e.g.
        ``"prompt_sending_airt_hate"``) used for resume tracking.  However,
        user-facing output (console printer, reports) often needs to
        aggregate results along a *different* dimension — for example,
        grouping by harm category rather than by technique.  The display
        group provides that second grouping axis without affecting resume
        behaviour.

        The default groups by technique name.  Subclasses override to
        change the aggregation axis:

        - **By technique** (default): ``return technique_name``
        - **By harm category / dataset**: ``return seed_group_name``
        - **Cross-product**: ``return f"{technique_name}_{seed_group_name}"``

        Note: ``seed_group_name`` is the dataset key from
        ``DatasetConfiguration.get_seed_attack_groups()`` (e.g.
        ``"airt_hate"``), not a ``SeedGroup`` object.

        Args:
            technique_name: The name of the attack technique.
            seed_group_name: The dataset key from the dataset configuration.

        Returns:
            str: The display-group label.
        """
        return technique_name

    def _get_default_objective_scorer(self) -> TrueFalseScorer:
        # Deferred import to avoid circular dependency.
        from pyrit.setup.initializers.components.scorers import ScorerInitializerTags

        # first check if the registry has a default objective scorer
        # if available either itself, or its chat target will be used
        chat_target: PromptTarget | None = None
        registry_default_scorer: TrueFalseScorer | None = None
        entries = ScorerRegistry.get_registry_singleton().get_by_tag(tag=ScorerInitializerTags.DEFAULT_OBJECTIVE_SCORER)
        if entries and isinstance(entries[0].instance, TrueFalseScorer):
            registry_default_scorer = entries[0].instance
            chat_target = registry_default_scorer.get_chat_target()
            logger.info(
                f"The registry contains default objective scorer: {type(registry_default_scorer).__name__} "
                f"with chat target: {type(chat_target).__name__ if chat_target else 'None'}"
            )

        chat_target = chat_target or get_default_scorer_target()

        # if the scenario has override composite scorer questions, use them to build a composite scorer
        composite_scorer_questions_paths = type(self)._get_additional_scoring_questions()
        if composite_scorer_questions_paths:
            path_scorers: list[TrueFalseScorer] = [
                SelfAskTrueFalseScorer(chat_target=chat_target, true_false_question_path=path)
                for path in composite_scorer_questions_paths
            ]
            backstop_scorer = TrueFalseInverterScorer(scorer=SelfAskRefusalScorer(chat_target=chat_target))
            scorer = TrueFalseCompositeScorer(
                aggregator=TrueFalseScoreAggregator.AND,
                scorers=[*path_scorers, backstop_scorer],
            )
            logger.info(
                f"Using composite default objective scorer: {type(scorer).__name__} "
                f"with chat target: {type(chat_target).__name__}"
            )
            return scorer

        if registry_default_scorer:
            logger.info(
                f"Using registry default objective scorer: {type(registry_default_scorer).__name__} "
                f"with chat target: {type(chat_target).__name__ if chat_target else 'None'}"
            )
            return registry_default_scorer

        scorer = TrueFalseInverterScorer(scorer=SelfAskRefusalScorer(chat_target=chat_target))
        logger.warning(
            f"Using fallback default objective scorer: {type(scorer).__name__} "
            f"with chat target: {type(chat_target).__name__ if chat_target else 'None'}"
        )
        return scorer

    def set_params_from_args(self, *, args: dict[str, Any]) -> None:
        """
        Populate ``self.params`` from merged CLI / config arguments.

        Coerces each value to its declared ``param_type``, validates, and
        materializes declared defaults for params not in ``args``. Every
        declared parameter is guaranteed a key in ``self.params`` after this
        call; params without a declared default land as ``None``.

        Args:
            args (dict[str, Any]): Map of parameter name to raw value. Keys
                with ``None`` values are treated as absent (YAML ``null``).
                Argparse callers should use ``argparse.SUPPRESS``.

        Raises:
            ValueError: Invalid declaration, unknown parameter, coercion
                failure, or value not in ``choices``.
        """
        declared = list(self.supported_parameters())
        if not self._declarations_validated:
            self._validate_declarations(declared=declared)
            self._declarations_validated = True

        declared_by_name = {p.name: p for p in declared}

        # None values are treated as absent so YAML `key: null` falls through to defaults.
        supplied = {name: value for name, value in args.items() if value is not None}

        coerced: dict[str, Any] = {}
        for name, raw_value in supplied.items():
            param = declared_by_name.get(name)
            if param is None:
                # Stash unknowns so _validate_params can list them all at once.
                coerced[name] = raw_value
                continue
            coerced[name] = coerce_value(param=param, raw_value=raw_value)

        self._validate_params(params=coerced, declared=declared)

        for param in declared:
            if param.name in coerced:
                continue
            # Materialize every declared param so scenarios can rely on
            # ``self.params[name]`` never raising ``KeyError``. Params declared
            # without an explicit default land as None, and the scenario raises
            # a domain-specific error at run time if it cannot proceed.
            coerced[param.name] = (
                copy.deepcopy(coerce_value(param=param, raw_value=param.default)) if param.default is not None else None
            )

        self.params = coerced

    def _validate_declarations(self, *, declared: list[Parameter]) -> None:
        """
        Validate the scenario's parameter declarations on first use.

        Args:
            declared (list[Parameter]): The ``supported_parameters()`` snapshot.

        Raises:
            ValueError: If declarations contain duplicate names, an
                unsupported ``param_type``, ``choices`` not coercible to
                ``param_type``, or a default that fails coercion / is not
                in ``choices``.
        """
        seen: set[str] = set()
        for param in declared:
            if param.name in seen:
                raise ValueError(f"Scenario '{type(self).__name__}' declares duplicate parameter name '{param.name}'.")
            seen.add(param.name)

            try:
                validate_param_type(param=param)
            except ValueError as exc:
                raise ValueError(f"Scenario '{type(self).__name__}' {exc}") from exc

            if param.choices is not None and get_origin(param.param_type) is list:
                # argparse `nargs='+'` applies choices per-item; core checks the whole list.
                # Reject the combination until we reconcile the semantics.
                raise ValueError(
                    f"Scenario '{type(self).__name__}' parameter '{param.name}' declares choices on a list "
                    f"param_type ({param.param_type!r}); this combination is not supported. "
                    f"Use a scalar param_type with choices, or omit choices on list params."
                )

            if param.choices is not None and param.param_type is not None:
                # Each choice must be coercible — fail at declaration time, not user time.
                for choice in param.choices:
                    try:
                        coerce_value(param=param, raw_value=choice)
                    except ValueError as exc:
                        raise ValueError(
                            f"Scenario '{type(self).__name__}' parameter '{param.name}' choice "
                            f"{choice!r} is not coercible to {param.param_type!r}: {exc}"
                        ) from exc

            if param.default is not None:
                try:
                    coerced_default = coerce_value(param=param, raw_value=param.default)
                except ValueError as exc:
                    raise ValueError(
                        f"Scenario '{type(self).__name__}' parameter '{param.name}' has an invalid default: {exc}"
                    ) from exc

                if param.choices is not None and coerced_default not in param.choices:
                    raise ValueError(
                        f"Scenario '{type(self).__name__}' parameter '{param.name}' default "
                        f"{param.default!r} is not in declared choices {param.choices!r}."
                    )

    def _validate_params(self, *, params: dict[str, Any], declared: list[Parameter]) -> None:
        """
        Validate supplied params against the scenario's declarations.

        Args:
            params (dict[str, Any]): Coerced (declared names) or raw (unknown) values.
            declared (list[Parameter]): Declarations snapshot from the caller, so
                the whole call sees one consistent view.

        Raises:
            ValueError: If any keys in ``params`` are not declared.
        """
        declared_names = {p.name for p in declared}

        unknown = sorted(set(params.keys()) - declared_names)
        if unknown:
            raise ValueError(
                f"Scenario '{type(self).__name__}' received unknown parameter(s): {', '.join(unknown)}. "
                f"Supported parameters: "
                f"{', '.join(sorted(declared_names)) if declared_names else 'none'}."
            )

    def _prepare_strategies(
        self,
        strategies: Optional[Sequence[ScenarioStrategy]],
    ) -> list[ScenarioStrategy]:
        """
        Resolve strategy inputs into a concrete list for this scenario.

        The default implementation calls resolve() on the strategy class, which handles
        None (use default), empty list (also use default), and aggregate expansion.

        Subclasses with complex composition semantics (e.g., RedTeamAgent with
        FoundryComposite) should override this to build their own composite types.

        Args:
            strategies: Strategy inputs from initialize_async. None or [] both mean use
                default; otherwise a list of strategies to resolve.

        Returns:
            list[ScenarioStrategy]: Ordered, deduplicated concrete strategies.
        """
        return self._strategy_class.resolve(strategies, default=self.get_default_strategy())

    @apply_defaults
    async def initialize_async(
        self,
        *,
        objective_target: PromptTarget = REQUIRED_VALUE,  # type: ignore[ty:invalid-parameter-default]
        scenario_strategies: Optional[Sequence[ScenarioStrategy]] = None,
        dataset_config: Optional[DatasetConfiguration] = None,
        max_concurrency: int = 10,
        max_retries: int = 0,
        memory_labels: Optional[dict[str, str]] = None,
        include_baseline: bool | None = None,
    ) -> None:
        """
        Initialize the scenario by populating self._atomic_attacks and creating the ScenarioResult.

        This method allows scenarios to be initialized with atomic attacks after construction,
        which is useful when atomic attacks require async operations to be built.

        If a scenario_result_id was provided in __init__, this method will check if it exists
        in memory and validate that the stored scenario matches the current configuration.
        If it matches, the scenario will resume from prior progress. If it doesn't match or
        doesn't exist, a new scenario result will be created.

        Args:
            objective_target (PromptTarget): The target system to attack.
            scenario_strategies (Optional[Sequence[ScenarioStrategy]]): The strategies to execute.
                Can be a list of ScenarioStrategy enum members. If None, uses the default aggregate
                from the scenario's configuration.
            dataset_config (Optional[DatasetConfiguration]): Configuration for the dataset source.
                Use this to specify dataset names or maximum dataset size from the CLI.
                If not provided, scenarios use their default_dataset_config().
            max_concurrency (int): Maximum number of concurrent attack executions. Defaults to 1.
            max_retries (int): Maximum number of automatic retries if the scenario raises an exception.
                Set to 0 (default) for no automatic retries. If set to a positive number,
                the scenario will automatically retry up to this many times after an exception.
                For example, max_retries=3 allows up to 4 total attempts (1 initial + 3 retries).
            memory_labels (Optional[Dict[str, str]]): Additional labels to apply to all
                attack runs in the scenario. These help track and categorize the scenario.
            include_baseline (bool | None): Whether to prepend a baseline atomic attack that sends
                all objectives without modifications, allowing comparison between unmodified prompts
                and the scenario's strategies. If None (the default), the scenario type's
                ``BASELINE_ATTACK_POLICY`` class attribute decides: ``Enabled`` includes it,
                ``Disabled`` omits it, and ``Forbidden`` always omits it (and rejects an
                explicit ``True``). Passing ``True`` to a scenario whose ``BASELINE_ATTACK_POLICY``
                is ``Forbidden`` raises ``ValueError``.

        Raises:
            ValueError: If no objective_target is provided, or if ``include_baseline=True`` is passed
                to a scenario whose ``BASELINE_ATTACK_POLICY`` is ``Forbidden``.
        """
        # Validate required parameters
        if objective_target is None:
            raise ValueError(
                "objective_target is required. "
                "Provide it either as a parameter or via set_default_value() in an initialization script."
            )

        # Set instance variables from parameters
        self._objective_target = objective_target
        self._objective_target_identifier = objective_target.get_identifier()
        type(self).TARGET_REQUIREMENTS.validate(target=objective_target)
        self._dataset_config_provided = dataset_config is not None
        self._dataset_config = dataset_config if dataset_config else self.default_dataset_config()
        self._max_concurrency = max_concurrency
        self._max_retries = max_retries
        self._memory_labels = memory_labels or {}

        # Deprecated. Will be removed in 0.16.0. Honor the legacy constructor-time
        # include_default_baseline (or subclass include_baseline) only when the caller did
        # not supply a runtime value.
        if include_baseline is None and self._legacy_include_baseline is not None:
            include_baseline = self._legacy_include_baseline

        # Resolve the effective include_baseline. Forbidden is checked first so a forbidden
        # scenario type never silently inherits a True default; explicit-True on a forbidden
        # type is a hard error rather than a silent ignore. For the Enabled / Disabled states,
        # a None runtime value defers to the policy.
        if self.BASELINE_ATTACK_POLICY is BaselineAttackPolicy.Forbidden:
            if include_baseline is True:
                raise ValueError(
                    f"{type(self).__name__} does not support a default baseline "
                    f"(BASELINE_ATTACK_POLICY = Forbidden); pass include_baseline=False or omit the argument."
                )
            include_baseline = False
        elif include_baseline is None:
            include_baseline = self.BASELINE_ATTACK_POLICY is BaselineAttackPolicy.Enabled

        self._include_baseline = include_baseline

        # Prepare scenario strategies using the stored configuration
        self._scenario_strategies = self._prepare_strategies(scenario_strategies)

        # Materialize declared defaults for programmatic callers that skip the
        # explicit set_params_from_args step. Frontend-driven flows already
        # call it (which sets _declarations_validated=True), so this is a no-op
        # in that path.
        if not self._declarations_validated:
            self.set_params_from_args(args={})

        self._atomic_attacks = await self._get_steps_async()

        # Deprecation rescue. Will be removed in 0.16.0. If the override didn't emit baseline,
        # warn and inject. Migrated overrides emit baseline themselves and bypass this branch.
        # Reuse seeds from the first existing attack rather than re-resolving from
        # dataset_config; re-resolution under max_dataset_size would draw a fresh sample
        # (the very ADO 9012 bug this PR fixes). When no atomic attacks exist yet the
        # rescue falls back to the dataset_config one-time resolution.
        if include_baseline and (not self._atomic_attacks or self._atomic_attacks[0].atomic_attack_name != "baseline"):
            print_deprecation_message(
                old_item=f"Implicit baseline injection for {type(self).__name__}._get_atomic_attacks_async()",
                new_item="explicit emission via self._build_baseline_atomic_attack(seed_groups=...) in the override",
                removed_in="0.16.0",
            )
            if self._atomic_attacks:
                seed_groups = self._atomic_attacks[0].seed_groups
            else:
                seed_groups = self._dataset_config.get_all_seed_attack_groups()
            self._atomic_attacks.insert(0, self._build_baseline_atomic_attack(seed_groups=seed_groups))

        # Snapshot params onto the identifier before the resume branch so the identifier
        # is fully populated regardless of which branch we take. Deep-copy avoids sharing
        # mutable state with self.params.
        params_snapshot = copy.deepcopy(self.params)
        _assert_json_serializable(params=params_snapshot)
        self._identifier.init_data = params_snapshot

        # Check if we're resuming an existing scenario. Any divergence is a hard error
        # rather than a silent restart, so the original progress isn't orphaned without
        # the user knowing.
        if self._scenario_result_id:
            existing_results = self._memory.get_scenario_results(scenario_result_ids=[self._scenario_result_id])

            if not existing_results:
                raise ValueError(
                    f"Scenario result id '{self._scenario_result_id}' not found in memory. "
                    f"Drop scenario_result_id to start a new scenario."
                )

            self._validate_stored_scenario(stored_result=existing_results[0])
            self._apply_persisted_objectives(stored_result=existing_results[0])
            return  # Valid resume - skip creating new scenario result

        # Build display group mapping from atomic attacks
        self._display_group_map = {aa.atomic_attack_name: aa.display_group for aa in self._atomic_attacks}

        # Create new scenario result
        attack_results: dict[str, list[AttackResult]] = {
            atomic_attack.atomic_attack_name: [] for atomic_attack in self._atomic_attacks
        }

        result = ScenarioResult(
            scenario_identifier=self._identifier,
            objective_target_identifier=self._objective_target_identifier,
            objective_scorer_identifier=self._objective_scorer_identifier,
            labels=self._memory_labels,
            attack_results=attack_results,
            scenario_run_state="CREATED",
            display_group_map=self._display_group_map,
            metadata=self._build_initial_scenario_metadata(),
        )

        self._memory.add_scenario_results_to_memory(scenario_results=[result])
        self._scenario_result_id = str(result.id)
        logger.info(f"Created new scenario result with ID: {self._scenario_result_id}")

    def _build_initial_scenario_metadata(self) -> dict[str, Any]:
        """
        Build the metadata dict persisted with a freshly-created ``ScenarioResult``.

        When ``max_dataset_size`` is in effect, the dataset config draws an
        unseeded ``random.sample`` and the chosen subset would silently change
        on the next run (e.g. a resume). To make resume reliable, snapshot the
        chosen objective hashes here so the next ``_setup_scenario_async`` can
        replay them via ``keep_seed_groups_with_hashes``.

        When ``max_dataset_size`` is not set, the sample equals the dataset and
        nothing needs pinning; the dict is empty.

        Returns:
            dict[str, Any]: Metadata payload for the new ScenarioResult.
        """
        metadata: dict[str, Any] = {}
        if getattr(self._dataset_config, "max_dataset_size", None) is None:
            return metadata
        hashes: list[str] = []
        seen: set[str] = set()
        for aa in self._atomic_attacks:
            for sg in aa.seed_groups:
                if sg.objective is None:
                    continue
                sha = to_sha256(sg.objective.value)
                if sha not in seen:
                    seen.add(sha)
                    hashes.append(sha)
        metadata["objective_hashes"] = hashes
        return metadata

    async def _finalize_scenario_result_async(self, *, scenario_result_id: str) -> None:
        """
        Persist any run-summary state to the scenario result before COMPLETED.

        Called once per successful execution attempt of ``_execute_scenario_async``,
        right after the final step completes and before the
        ``update_scenario_run_state(COMPLETED)`` transition lands. Subclasses
        that need to record run-summary state (e.g. composition pipelines
        writing per-phase outcomes into ``ScenarioResult.metadata``) should
        override this method.

        The default is a no-op. The ``scenario_result_id`` is supplied so
        subclasses don't need to re-derive it from ``self._scenario_result_id``.

        Args:
            scenario_result_id (str): The id of the scenario result that is
                about to be marked COMPLETED. Use
                ``self._memory.update_scenario_metadata`` to write into it.
        """
        return

    def _apply_persisted_objectives(self, *, stored_result: ScenarioResult) -> None:
        """
        On resume, replay the originally-sampled objective subset.

        When the first run used ``max_dataset_size``, the chosen subset was
        recorded in ``ScenarioResult.metadata["objective_hashes"]``.
        Restrict each atomic attack's freshly-resolved seed_groups to that set
        so a fresh ``random.sample`` draw on resume can't silently shift which
        objectives the scenario operates on. If any persisted hash is no longer
        present in the dataset, refuse to resume — running a smaller subset
        than the user committed to would silently produce different results.

        Args:
            stored_result (ScenarioResult): The scenario result loaded from memory.

        Raises:
            ValueError: If any persisted objective hash is missing from the
                currently-resolved dataset.
        """
        metadata = stored_result.metadata or {}
        persisted = metadata.get("objective_hashes")
        if not persisted:
            return

        persisted_hashes: set[str] = set(persisted)
        retained: set[str] = set()
        for aa in self._atomic_attacks:
            retained |= aa.keep_seed_groups_with_hashes(hashes=persisted_hashes)

        missing = persisted_hashes - retained
        if missing:
            sample = sorted(missing)[:3]
            raise ValueError(
                f"Scenario result id '{self._scenario_result_id}' cannot resume: "
                f"{len(missing)} persisted objective hash(es) are no longer present in the dataset "
                f"(missing examples: {', '.join(h[:12] + '...' for h in sample)}). "
                f"Either restore the missing objectives or drop scenario_result_id to start a new scenario."
            )

    def _build_baseline_atomic_attack(self, *, seed_groups: list[SeedAttackGroup]) -> AtomicAttack:
        """
        Build the baseline AtomicAttack from pre-resolved seed groups.

        The baseline sends each objective unmodified, providing a comparison point
        against the scenario's strategy attacks. Pass the same ``seed_groups`` used
        to build the strategy attacks so both populations match.

        Args:
            seed_groups: Seed groups to attack. Used as-is, no further sampling.

        Returns:
            AtomicAttack: The baseline atomic attack.

        Raises:
            ValueError: If ``initialize_async`` has not been called (no objective
                target or scorer set).
        """
        if self._objective_target is None:
            raise ValueError("Objective target is required to create baseline attack.")
        if self._objective_scorer is None:
            raise ValueError("Objective scorer is required to create baseline attack.")

        from pyrit.executor.attack.core.attack_config import AttackScoringConfig

        attack = PromptSendingAttack(
            objective_target=self._objective_target,
            attack_scoring_config=AttackScoringConfig(objective_scorer=cast("TrueFalseScorer", self._objective_scorer)),
        )

        return AtomicAttack(
            atomic_attack_name="baseline",
            attack_technique=AttackTechnique(attack=attack),
            seed_groups=seed_groups,
            memory_labels=self._memory_labels,
        )

    def _raise_dataset_exception(self) -> None:
        error_msg = textwrap.dedent(
            f"""
            Dataset is not available or failed to load.
            Scenarios require datasets loaded in CentralMemory or to be passed explicitly.
            Either load the datasets into the database before running the scenario, or for
            example datasets, you can use the `load_default_datasets` initializer.

            Required datasets: {", ".join(self.default_dataset_config().get_default_dataset_names())}
            """
        )
        raise ValueError(error_msg)

    def _validate_stored_scenario(self, *, stored_result: ScenarioResult) -> None:
        """
        Validate that a stored scenario result exactly matches the current scenario configuration.

        Resume is opt-in via ``scenario_result_id``; any divergence from the stored
        result is treated as user error rather than a silent restart, since the
        original progress would otherwise be orphaned without warning.

        Args:
            stored_result (ScenarioResult): The scenario result retrieved from memory.

        Raises:
            ValueError: If the stored scenario name, version, or parameters do not
                match the current configuration.
        """
        stored_name = stored_result.scenario_identifier.name
        stored_version = stored_result.scenario_identifier.version

        if stored_name != self._identifier.name:
            raise ValueError(
                f"Scenario result id '{self._scenario_result_id}' belongs to scenario '{stored_name}' "
                f"but current scenario is '{self._identifier.name}'. "
                f"Drop scenario_result_id to start a new scenario."
            )

        if stored_version != self._identifier.version:
            raise ValueError(
                f"Scenario result id '{self._scenario_result_id}' was created with "
                f"{self._identifier.name} version {stored_version} but current version is "
                f"{self._identifier.version}. Drop scenario_result_id to start a new scenario."
            )

        # Treat None (legacy result without persisted params) as empty. Compare both sides
        # post-JSON-roundtrip so types that the memory column rewrites (tuple → list, non-str
        # dict keys → str) don't surface as false mismatches under param_type=None.
        stored_params = stored_result.scenario_identifier.init_data or {}
        current_params_normalized = json.loads(json.dumps(self.params))
        if stored_params != current_params_normalized:
            diff = _format_param_key_diff(stored=stored_params, current=current_params_normalized)
            raise ValueError(
                f"Scenario result id '{self._scenario_result_id}' has mismatched parameters ({diff}). "
                f"Drop scenario_result_id to start a new scenario, or pass matching parameters to resume."
            )

        logger.info(
            f"Resuming scenario '{self._name}' from existing result "
            f"(ID: {self._scenario_result_id}, state: {stored_result.scenario_run_state})"
        )

    def _get_completed_objective_hashes_for_attack(self, *, atomic_attack: AtomicAttack) -> set[str]:
        """
        Return the set of ``objective_sha256`` values already completed (non-error)
        for a specific atomic attack inside this scenario.

        Queries ``AttackResultEntry`` rows directly by ``attribution_parent_id`` —
        which is stamped at write-time by the attack persistence path — so
        results from an interrupted run are visible even though the
        ``ScenarioResult.attack_results`` aggregate may not yet reflect them.
        Identity is content-derived (``to_sha256(objective)``), so it stays
        stable even if ``get_seed_groups()`` reorders or resamples between runs.

        Rows are matched on ``(parent_collection, parent_eval_hash)`` so that
        two ``AtomicAttack`` instances sharing a name but using different
        techniques (e.g. base64 vs hex encoders) never cross-pollinate their
        completed-hash sets on resume. Rows persisted before
        ``parent_eval_hash`` was introduced (or by callers that don't supply
        one) match name-only as a backward-compatible fallback.

        Args:
            atomic_attack (AtomicAttack): The live atomic attack whose
                ``atomic_attack_name`` and technique identifier scope the query.

        Returns:
            set[str]: ``objective_sha256`` hex strings for completed-without-error rows.
        """
        if not self._scenario_result_id:
            return set()

        atomic_attack_name = atomic_attack.atomic_attack_name
        expected_eval_hash = atomic_attack.technique_eval_hash

        completed_hashes: set[str] = set()
        try:
            rows = self._memory.get_attack_results(scenario_result_id=self._scenario_result_id)
            for row in rows:
                if row.outcome == AttackOutcome.ERROR:
                    continue
                if row.attribution_data is None:
                    continue
                if row.attribution_data.get("parent_collection") != atomic_attack_name:
                    continue
                row_eval_hash = row.attribution_data.get("parent_eval_hash")
                if row_eval_hash is not None and row_eval_hash != expected_eval_hash:
                    continue
                if row.objective:
                    completed_hashes.add(to_sha256(row.objective))
        except Exception as e:
            logger.warning(
                f"Failed to retrieve completed objective hashes for atomic attack '{atomic_attack_name}': {str(e)}"
            )

        return completed_hashes

    async def _get_remaining_atomic_attacks_async(self) -> list[AtomicAttack]:
        """
        Get the list of atomic attacks that still have objectives to complete.

        Uses ``objective_sha256`` as the stable identity for resume: each
        atomic attack enforces uniqueness of objective hashes at construction
        time, and the executor stamps ``attribution_parent_id`` +
        ``attribution_data["parent_collection"]`` on the row so a content-hash
        join is sufficient.

        Returns:
            List[AtomicAttack]: List of atomic attacks with uncompleted objectives.
        """
        if not self._scenario_result_id:
            # No scenario result yet, return all atomic attacks
            return self._atomic_attacks

        remaining_attacks: list[AtomicAttack] = []

        for atomic_attack in self._atomic_attacks:
            completed_hashes = self._get_completed_objective_hashes_for_attack(atomic_attack=atomic_attack)

            if completed_hashes:
                original_count = len(atomic_attack.seed_groups)
                atomic_attack.drop_seed_groups_with_hashes(hashes=completed_hashes)
                remaining_count = len(atomic_attack.seed_groups)
                if remaining_count == 0:
                    logger.info(
                        f"Atomic attack '{atomic_attack.atomic_attack_name}' has all objectives completed, skipping"
                    )
                    continue
                if remaining_count < original_count:
                    logger.info(
                        f"Atomic attack '{atomic_attack.atomic_attack_name}' has "
                        f"{remaining_count}/{original_count} objectives remaining"
                    )

            remaining_attacks.append(atomic_attack)

        return remaining_attacks

    async def _get_steps_async(self) -> list[AtomicAttack]:
        """
        Build the steps this scenario will execute.

        Returns the list of :class:`AtomicAttack` instances the orchestrator
        walks via the default linear policy. Subclasses override this method
        to author custom step inventories — adaptive selectors, hand-rolled
        composites, or wrappers around the registry pattern.

        The default implementation builds atomic attacks from the cross-product
        of selected techniques and datasets. Uses
        ``_get_attack_technique_factories()`` to obtain factories, then iterates
        over every (technique, dataset) pair to create an ``AtomicAttack`` for
        each. Grouping for display is controlled by ``_build_display_group()``.

        For backward compatibility, subclasses that still override
        :meth:`_get_atomic_attacks_async` are detected automatically and routed
        through that override; a deprecation warning is emitted once per such
        subclass at class-creation time. Removal of the old method is planned
        for ``0.16.0``.

        Subclasses that do **not** use the factory/registry pattern should
        override this method entirely. Overrides that want baseline support
        must call ``self._build_baseline_atomic_attack`` with the strategy
        seeds.

        Returns:
            list[AtomicAttack]: The generated steps.

        Raises:
            ValueError: If the scenario has not been initialized.
        """
        # Legacy-override delegation: if a subclass still overrides the old
        # name (and didn't also override _get_steps_async), call that override
        # so we don't lose its behavior during the deprecation window.
        if type(self)._get_atomic_attacks_async is not Scenario._get_atomic_attacks_async:
            return await self._get_atomic_attacks_async()

        if self._objective_target is None:
            raise ValueError(
                "Scenario not properly initialized. Call await scenario.initialize_async() before running."
            )

        from pyrit.executor.attack import AttackScoringConfig

        selected_techniques = {s.value for s in self._scenario_strategies}

        factories = self._get_attack_technique_factories()
        seed_groups_by_dataset = self._dataset_config.get_seed_attack_groups()

        scoring_config = AttackScoringConfig(objective_scorer=cast("TrueFalseScorer", self._objective_scorer))

        atomic_attacks: list[AtomicAttack] = []
        for technique_name in selected_techniques:
            factory = factories.get(technique_name)
            if factory is None:
                logger.warning(f"No factory for technique '{technique_name}', skipping.")
                continue

            for dataset_name, seed_groups in seed_groups_by_dataset.items():
                if factory.seed_technique is not None:
                    compatible_groups = SeedAttackGroup.filter_compatible(
                        seed_groups=seed_groups,
                        technique=factory.seed_technique,
                    )
                    skipped = len(seed_groups) - len(compatible_groups)
                    if skipped:
                        logger.info(
                            f"Skipped {skipped} seed group(s) from '{dataset_name}' for technique "
                            f"'{technique_name}' (prompt sequences overlap with simulated conversation)."
                        )
                    if not compatible_groups:
                        logger.warning(
                            f"No compatible seed groups in '{dataset_name}' for technique "
                            f"'{technique_name}', skipping this (technique, dataset) pair."
                        )
                        continue
                else:
                    compatible_groups = list(seed_groups)

                attack_technique = factory.create(
                    objective_target=self._objective_target,
                    attack_scoring_config=scoring_config,
                )
                display_group = self._build_display_group(
                    technique_name=technique_name,
                    seed_group_name=dataset_name,
                )
                atomic_attacks.append(
                    AtomicAttack(
                        atomic_attack_name=f"{technique_name}_{dataset_name}",
                        attack_technique=attack_technique,
                        seed_groups=list(compatible_groups),
                        adversarial_chat=factory.adversarial_chat,
                        objective_scorer=cast("TrueFalseScorer", self._objective_scorer),
                        memory_labels=self._memory_labels,
                        display_group=display_group,
                    )
                )

        if self._include_baseline:
            all_seed_groups = [g for groups in seed_groups_by_dataset.values() for g in groups]
            atomic_attacks.insert(0, self._build_baseline_atomic_attack(seed_groups=all_seed_groups))

        return atomic_attacks

    async def _get_atomic_attacks_async(self) -> list[AtomicAttack]:
        """
        Delegate to :meth:`_get_steps_async`.

        Kept as a passthrough so existing subclass overrides keep working
        through the deprecation window. New scenarios should override
        :meth:`_get_steps_async` directly. Will be removed in ``0.16.0``.

        Returns:
            list[AtomicAttack]: Delegates to :meth:`_get_steps_async`.
        """
        return await self._get_steps_async()

    def _build_execution_graph(
        self, *, steps: Optional[Sequence[ScenarioStep]] = None
    ) -> StrategyGraph[ScenarioStep, int]:
        """
        Build the ``StrategyGraph`` that drives this execution attempt.

        Default implementation wraps the supplied ``steps`` (or, if omitted,
        ``self._atomic_attacks``) in a linear policy via
        ``_build_default_linear_policy``. This produces a graph whose traversal
        is identical to the legacy flat ``_execute_scenario_async`` loop, so
        scenarios that haven't opted into a richer policy see no behavior
        change.

        Subclasses with a state-machine flavor (rapid-response, adaptive,
        branching) override this to author their own ``StrategyPolicy`` and
        pass it to ``StrategyGraph``. Such overrides should still consume
        ``self._atomic_attacks`` as the seed of their step inventory so the
        existing resume-by-name path keeps working through Phase 5.

        Args:
            steps (Optional[Sequence[ScenarioStep]]): Steps to drive. ``None``
                falls back to ``self._atomic_attacks``. ``_execute_scenario_async``
                passes the resume-filtered list explicitly so already-completed
                steps are not re-executed.

        Returns:
            StrategyGraph[ScenarioStep, int]: The graph that ``run_async``
                will iterate.

        Raises:
            ValueError: If ``steps`` is empty (or unset and there are no
                atomic attacks).
        """
        effective_steps = list(steps) if steps is not None else list(self._atomic_attacks)
        if not effective_steps:
            raise ValueError(
                "Cannot build an execution graph with no steps. Either initialize the "
                "scenario via ``await scenario.initialize_async(...)`` so atomic attacks are "
                "populated, or override ``_build_execution_graph`` to supply your own steps."
            )
        return StrategyGraph(policy=self._build_default_linear_policy(steps=effective_steps))

    def _build_default_linear_policy(self, *, steps: Sequence[ScenarioStep]) -> StrategyPolicy[ScenarioStep, int]:
        """
        Build a linear-traversal policy that preserves scenario-level execution params.

        Each policy action runs ``steps[i].process_async()`` and transitions
        to state ``i + 1``; state ``len(steps)`` is the sole terminal state.
        Every step type — ``AtomicAttack``, ``AdaptiveStep``, or any future
        custom ``ScenarioStep`` subclass — flows through the same uniform
        dispatch path. ``AtomicAttack`` steps receive scenario-level
        ``max_concurrency`` via :meth:`AtomicAttack.set_scenario_max_concurrency`
        before the policy is frozen, so the unified action body does not need
        to branch on step type. The step's ``name`` is stamped into
        ``ScenarioStepResult.metadata['step_name']`` so the orchestrator can
        identify the step at yield time (caller-supplied metadata wins on
        collision).

        Args:
            steps (Sequence[ScenarioStep]): The steps to wrap. Must be non-empty.

        Returns:
            StrategyPolicy[ScenarioStep, int]: A frozen linear policy.

        Raises:
            ValueError: If ``steps`` is empty.
        """
        if not steps:
            raise ValueError("_build_default_linear_policy requires at least one step.")

        # Push the scenario-level max_concurrency into every AtomicAttack step
        # exactly once, before any action runs. Non-AtomicAttack steps either
        # own their own concurrency (e.g. AdaptiveStep) or default to 1, so the
        # orchestrator stays out of their dispatch.
        for step in steps:
            if isinstance(step, AtomicAttack):
                step.set_scenario_max_concurrency(self._max_concurrency)

        terminal_state = len(steps)
        actions: dict[int, PolicyAction[ScenarioStep, int]] = {}

        for index, step in enumerate(steps):

            async def _action(
                graph: StrategyGraph[ScenarioStep, int],
                _step: ScenarioStep = step,
                _next: int = index + 1,
            ) -> tuple[int, ScenarioStepResult | None]:
                graph.bind_active_steps(steps=(_step,))
                try:
                    base_result = await _step.process_async()
                    # Stamp ``step_name`` so the orchestrator can route the
                    # result without depending on ``graph.active_steps``
                    # (cleared before yield). Caller metadata wins on
                    # collision so steps remain authoritative.
                    merged_metadata = {"step_name": _step.name, **base_result.metadata}
                    result: ScenarioStepResult | None = ScenarioStepResult(
                        outcome=base_result.outcome,
                        attack_results=list(base_result.attack_results),
                        step_identifier=base_result.step_identifier,
                        metadata=merged_metadata,
                    )
                finally:
                    graph.bind_active_steps(steps=())
                return _next, result

            actions[index] = _action

        return StrategyPolicy(
            actions=actions,
            initial_state=0,
            terminal_states=frozenset({terminal_state}),
        )

    async def run_async(self) -> ScenarioResult:
        """
        Execute the scenario by walking its ``StrategyGraph``.

        Each ``ScenarioStep`` produces a ``ScenarioStepResult`` whose attack
        results are persisted in order and tagged with a ``step_identifier``
        so step-level filtering and grouping work alongside the existing
        ``atomic_attack_identifier`` lineage. The default execution graph
        produced by ``_build_execution_graph`` is a linear traversal of
        ``self._atomic_attacks``, so scenarios that have not opted into a
        richer policy see the same end-to-end behavior as before.

        The graph is rebuilt at the start of every execution attempt from the
        resume-filtered step list, so calling ``run_async`` after a partial
        failure skips already-completed work the same way the legacy flat
        loop did. ``self.execution_graph`` and ``self.execution_history``
        expose the current attempt's state.

        If ``max_retries`` is set, the scenario will automatically retry after
        an exception up to the specified number of times. Each retry rebuilds
        the graph from the current remaining steps.

        Returns:
            ScenarioResult: Contains scenario identifier and aggregated list of
                attack results from every step that ran.

        Raises:
            ValueError: If the scenario has no atomic attacks configured. If your
                scenario requires initialization, call
                ``await scenario.initialize_async()`` first.
            ValueError: If the scenario raises an exception after exhausting all
                retry attempts.
            RuntimeError: If the scenario fails for any other reason while
                executing.

        Example:
            >>> result = await scenario.run_async()
            >>> print(f"Scenario: {result.scenario_identifier.name}")
            >>> print(f"Total results: {len(result.attack_results)}")
        """
        if not self._atomic_attacks:
            raise ValueError(
                "Cannot run scenario with no atomic attacks. Either supply them in initialization or "
                "call await scenario.initialize_async() first."
            )

        if not self._scenario_result_id:
            raise ValueError("Scenario not properly initialized. Call await scenario.initialize_async() first.")

        # Type narrowing: create local variable that type checker knows is non-None
        scenario_result_id: str = self._scenario_result_id

        # Implement retry logic
        last_exception = None
        for retry_attempt in range(self._max_retries + 1):  # +1 for initial attempt
            try:
                return await self._execute_scenario_async()
            except Exception as e:
                last_exception = e

                # Get current scenario to check number of tries
                scenario_results = self._memory.get_scenario_results(scenario_result_ids=[scenario_result_id])
                current_tries = scenario_results[0].number_tries if scenario_results else retry_attempt + 1

                # Check if we have more retries available
                remaining_retries = self._max_retries - retry_attempt

                if remaining_retries > 0:
                    logger.error(
                        f"Scenario '{self._name}' failed on attempt {current_tries} with error: {str(e)}. "
                        f"Retrying... ({remaining_retries} retries remaining)",
                        exc_info=True,
                    )
                    # Continue to next iteration for retry
                    continue
                # No more retries, log final failure
                logger.error(
                    f"Scenario '{self._name}' failed after {current_tries} attempts "
                    f"(initial + {self._max_retries} retries) with error: {str(e)}. Giving up.",
                    exc_info=True,
                )
                raise

        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError(f"Scenario '{self._name}' completed unexpectedly without result")

    async def _execute_scenario_async(self) -> ScenarioResult:
        """
        Perform a single execution attempt of the scenario.

        Iterates ``self.execution_graph.event_loop_async()`` and applies the
        same per-step persistence, partial-failure handling, and retry
        semantics that the legacy flat loop applied per-``AtomicAttack``. The
        graph is built once per execution attempt from the resume-filtered
        ``self._atomic_attacks`` so already-completed steps are skipped.

        Returns:
            ScenarioResult: The result of this execution attempt.

        Raises:
            ValueError: If ``self._scenario_result_id`` is missing or any
                step partially fails.
            Exception: Any exception raised while executing a step is logged,
                the scenario is marked ``FAILED``, and the exception is re-raised.
        """
        # Lazy import to avoid module-level circularity: build_step_identifier
        # lives in pyrit.identifiers which itself imports several pyrit.models
        # types that the scenario module re-exports indirectly.
        from pyrit.identifiers.evaluation_identifier import StepEvaluationIdentifier
        from pyrit.identifiers.step_identifier import build_step_identifier
        from pyrit.memory.memory_models import MAX_IDENTIFIER_VALUE_LENGTH

        logger.info(f"Starting scenario '{self._name}' execution with {len(self._atomic_attacks)} atomic attacks")

        # Type narrowing: _scenario_result_id is guaranteed to be non-None at this point
        # (verified in run_async before calling this method)
        if self._scenario_result_id is None:
            raise ValueError("self._scenario_result_id is not initialized")
        scenario_result_id: str = self._scenario_result_id

        # Increment number_tries at the start of each run
        scenario_results = self._memory.get_scenario_results(scenario_result_ids=[scenario_result_id])
        if scenario_results:
            current_scenario = scenario_results[0]
            current_scenario.number_tries += 1
            entry = ScenarioResultEntry(entry=current_scenario)
            self._memory._update_entry(entry)
            logger.info(f"Scenario '{self._name}' attempt #{current_scenario.number_tries}")
        else:
            raise ValueError(f"Scenario result with ID {scenario_result_id} not found")

        # Get remaining atomic attacks (filters out completed ones and updates objectives)
        remaining_attacks = await self._get_remaining_atomic_attacks_async()

        if not remaining_attacks:
            logger.info(f"Scenario '{self._name}' has no remaining objectives to execute")
            # Mark scenario as completed
            self._memory.update_scenario_run_state(
                scenario_result_id=scenario_result_id, scenario_run_state="COMPLETED"
            )
            # Retrieve and return the current scenario result
            scenario_results = self._memory.get_scenario_results(scenario_result_ids=[scenario_result_id])
            if scenario_results:
                return scenario_results[0]
            raise ValueError(f"Scenario result with ID {scenario_result_id} not found")

        logger.info(
            f"Scenario '{self._name}' has {len(remaining_attacks)} atomic attacks "
            f"with remaining objectives (out of {len(self._atomic_attacks)} total)"
        )

        # Mark scenario as in progress
        self._memory.update_scenario_run_state(scenario_result_id=scenario_result_id, scenario_run_state="IN_PROGRESS")

        # Stamp scenario_result_id on every AtomicAttack step BEFORE building the execution
        # graph so policy closures see the attribution when they call step.run_async(). The
        # attribution_parent_id foreign key on each persisted AttackResult row is set by the
        # attack event handler at write time (no post-atomic bulk manifest write needed).
        # Non-AtomicAttack ScenarioStep subclasses opt in by exposing set_scenario_result_id.
        for _step in remaining_attacks:
            if isinstance(_step, AtomicAttack):
                _step.set_scenario_result_id(scenario_result_id)

        # Build a fresh execution graph from the resume-filtered steps for this attempt.
        # We always rebuild on retry so the policy reflects the currently-pending work.
        self._execution_graph = self._build_execution_graph(steps=remaining_attacks)

        # Calculate starting index based on completed attacks (for progress bar continuity).
        total_steps = len(self._atomic_attacks)
        completed_count = total_steps - len(remaining_attacks)
        progress = tqdm(
            desc=f"Executing {self._name}",
            unit="attack",
            total=total_steps,
            initial=completed_count,
        )
        step_position = completed_count
        # Track the most recent step we attempted so a step-raised exception
        # can still log the offending step's name. ``graph.active_steps`` is
        # cleared in the policy action's ``finally`` before the exception
        # propagates, so it's not a reliable post-mortem source.
        last_attempted_step_name: str = "<unknown_step>"

        try:
            try:
                async for step_result in self._execution_graph.event_loop_async():
                    step_position += 1
                    step_name = step_result.metadata.get("step_name", "<unknown_step>")
                    last_attempted_step_name = step_name

                    logger.info(
                        f"Atomic attack {step_position}/{total_steps} ('{step_name}') in scenario '{self._name}'"
                    )

                    # Stamp step_identifier on every attack_result that doesn't already carry one.
                    # Steps may opt into setting it themselves (e.g., adaptive scenarios with
                    # nested attack executions); otherwise the default linear path stamps a
                    # one-attack-per-step composite identifier here. We mirror
                    # ``AtomicAttack._enrich_atomic_attack_identifiers``: populate the eval_hash
                    # before truncation so it survives the DB round-trip, then push the enriched
                    # identifier back to the AttackResultEntry row by attack_result_id.
                    for attack_result in step_result.attack_results:
                        if attack_result.step_identifier is None and attack_result.atomic_attack_identifier is not None:
                            new_identifier = build_step_identifier(
                                step_name=step_name,
                                outcome=step_result.outcome,
                                attack_execution_identifiers=[attack_result.atomic_attack_identifier],
                            )
                            if new_identifier.eval_hash is None:
                                new_identifier = new_identifier.with_eval_hash(
                                    StepEvaluationIdentifier(new_identifier).eval_hash
                                )
                            attack_result.step_identifier = new_identifier

                        # Push the (newly-stamped or pre-stamped) step_identifier to the existing
                        # AttackResultEntry so downstream ``get_scenario_results`` rehydrates it.
                        if attack_result.step_identifier is not None and attack_result.attack_result_id:
                            self._memory.update_attack_result_by_id(
                                attack_result_id=attack_result.attack_result_id,
                                update_fields={
                                    "step_identifier": attack_result.step_identifier.to_dict(
                                        max_value_length=MAX_IDENTIFIER_VALUE_LENGTH,
                                    ),
                                },
                            )

                    # Per-result scenario linkage is now stamped by the attack event handler
                    # at write time via AttackResultAttribution (set on each AtomicAttack by
                    # set_scenario_result_id above). No post-step bulk manifest update needed.

                    # Partial-failure handling. Only the AtomicAttack adapter path stuffs
                    # ``incomplete_objectives`` into metadata today; custom ScenarioStep
                    # subclasses opt in by populating the same key, so the same FAILED-state
                    # path covers any future step that wants partial-failure semantics.
                    incomplete_objectives = step_result.metadata.get("incomplete_objectives") or []
                    if incomplete_objectives:
                        incomplete_count = len(incomplete_objectives)
                        completed_in_step = len(step_result.attack_results)

                        logger.error(
                            f"Atomic attack {step_position}/{total_steps} "
                            f"('{step_name}') partially completed: "
                            f"{completed_in_step} completed, {incomplete_count} incomplete"
                        )

                        for obj, exc in incomplete_objectives:
                            logger.error(f"  Incomplete objective '{obj[:50]}...': {str(exc)}")

                        # Error AttackResults are linked to this scenario via the
                        # attribution_parent_id foreign key on AttackResultEntry (stamped by
                        # the attack event handler when an AttackResultAttribution is on the
                        # context). The previous per-scenario error_id manifest is no longer
                        # needed.

                        error_msg = (
                            f"Atomic attack '{step_name}' partially failed: "
                            f"{incomplete_count} of {incomplete_count + completed_in_step} "
                            f"objectives incomplete. See attack results for details."
                        )
                        self._memory.update_scenario_run_state(
                            scenario_result_id=scenario_result_id,
                            scenario_run_state="FAILED",
                            error_message=error_msg,
                            error_type=type(incomplete_objectives[0][1]).__name__,
                        )

                        raise ValueError(error_msg) from incomplete_objectives[0][1]

                    logger.info(
                        f"Atomic attack {step_position}/{total_steps} completed successfully with "
                        f"{len(step_result.attack_results)} results"
                    )
                    progress.update(1)

            except Exception as e:
                logger.error(
                    f"Atomic attack {step_position}/{total_steps} "
                    f"('{last_attempted_step_name}') failed in scenario '{self._name}': {str(e)}"
                )

                # Mark scenario as failed if not already done
                scenario_results = self._memory.get_scenario_results(scenario_result_ids=[scenario_result_id])
                if scenario_results and scenario_results[0].scenario_run_state != "FAILED":
                    self._memory.update_scenario_run_state(
                        scenario_result_id=scenario_result_id,
                        scenario_run_state="FAILED",
                        error_message=str(e),
                        error_type=type(e).__name__,
                    )

                raise

            logger.info(f"Scenario '{self._name}' completed successfully")

            # Give subclasses a chance to persist run-summary state on the
            # ScenarioResult (e.g. composition pipelines writing per-phase
            # outcomes into metadata) just before the COMPLETED transition.
            await self._finalize_scenario_result_async(scenario_result_id=scenario_result_id)

            # Mark scenario as completed
            self._memory.update_scenario_run_state(
                scenario_result_id=scenario_result_id, scenario_run_state="COMPLETED"
            )

            # Retrieve and return final scenario result
            scenario_results = self._memory.get_scenario_results(scenario_result_ids=[scenario_result_id])
            if not scenario_results:
                raise ValueError(f"Scenario result with ID {self._scenario_result_id} not found")

            return scenario_results[0]

        except Exception as e:
            logger.error(f"Scenario '{self._name}' failed with error: {str(e)}")
            raise
        finally:
            progress.close()
