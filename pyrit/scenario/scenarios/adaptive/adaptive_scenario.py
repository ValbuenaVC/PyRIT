# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``AdaptiveScenario`` — modality-agnostic base for scenarios that pick attack
techniques per-objective using an ``AdaptiveTechniqueSelector``.

Owns selector wiring, dispatcher construction, per-objective atomic-attack
emission, and resume rehydration. Concrete subclasses (``TextAdaptive``,
future ``ImageAdaptive`` / ``AudioAdaptive``) only declare strategy class,
default datasets, version, and atomic-attack prefix.

Baseline policy is ``Forbidden``: ``prompt_sending`` participates as one of
the selector's techniques rather than being prepended.
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import TYPE_CHECKING, ClassVar, cast

from pyrit.executor.attack import AttackScoringConfig
from pyrit.scenario.core.input_schema import RoleDescriptor, RoleTag
from pyrit.scenario.core.scenario import BaselineAttackPolicy, Scenario
from pyrit.scenario.core.scenario_step import ScenarioStep, ScenarioStepResult
from pyrit.scenario.core.strategy_graph import PolicyAction, StrategyGraph, StrategyPolicy
from pyrit.scenario.scenarios.adaptive.adaptive_step import AdaptiveStep
from pyrit.scenario.scenarios.adaptive.dispatcher import (
    ADAPTIVE_CONTEXT_LABEL,
    TechniqueBundle,
)
from pyrit.scenario.scenarios.adaptive.selector import (
    AdaptiveTechniqueSelector,
    ContextExtractor,
    global_context,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyrit.models import SeedAttackGroup
    from pyrit.prompt_target import PromptTarget
    from pyrit.scenario.core.atomic_attack import AtomicAttack
    from pyrit.score import TrueFalseScorer

logger = logging.getLogger(__name__)


class AdaptiveScenario(Scenario):
    """
    Abstract base for adaptive (epsilon-greedy) scenarios.

    Subclasses must implement the standard ``Scenario`` class-method overrides
    and declare ``VERSION`` and ``_atomic_attack_prefix``. Selector wiring,
    dispatcher construction, per-objective atomic-attack emission, and resume
    rehydration are handled here.
    """

    BASELINE_ATTACK_POLICY: ClassVar[BaselineAttackPolicy] = BaselineAttackPolicy.Forbidden

    #: Subclasses must declare a scenario version for memory bookkeeping.
    VERSION: ClassVar[int]

    #: Prefix for per-objective atomic-attack names (e.g. ``"adaptive_text"``).
    _atomic_attack_prefix: ClassVar[str] = "adaptive"

    def __init__(
        self,
        *,
        objective_scorer: TrueFalseScorer | None = None,
        epsilon: float = 0.2,
        pool_threshold: int = 3,
        max_attempts_per_objective: int = 3,
        seed: int | None = None,
        context_extractor: ContextExtractor = global_context,
        scenario_result_id: str | None = None,
    ) -> None:
        """
        Args:
            objective_scorer (TrueFalseScorer | None): Scorer used to judge each
                response. Defaults to the composite scorer from the base class.
            epsilon (float): Exploration probability for the selector. Defaults to 0.2.
            pool_threshold (int): Minimum per-(context, technique) attempts before
                the local estimate overrides the pooled rate. Set to 1 to disable
                pooling. Defaults to 3.
            max_attempts_per_objective (int): Max techniques per objective. Defaults to 3.
            seed (int | None): RNG seed for deterministic selection. Defaults to ``None``.
            context_extractor (ContextExtractor): Maps a ``SeedAttackGroup`` to a
                context key. Defaults to ``global_context``.
            scenario_result_id (str | None): ID of an existing ``ScenarioResult`` to resume.
        """
        if not objective_scorer:
            objective_scorer = self._get_default_objective_scorer()
        self._objective_scorer: TrueFalseScorer = objective_scorer

        self._epsilon = epsilon
        self._pool_threshold = pool_threshold
        self._max_attempts_per_objective = max_attempts_per_objective
        self._seed = seed
        self._context_extractor = context_extractor
        # Populated by _get_atomic_attacks_async; consumed by _build_execution_graph
        # only when an override path needs to introspect it externally.
        self._selector: AdaptiveTechniqueSelector | None = None

        super().__init__(
            version=self.VERSION,
            strategy_class=self.get_strategy_class(),
            objective_scorer=objective_scorer,
            scenario_result_id=scenario_result_id,
        )

    @classmethod
    def input_schema(cls) -> list[RoleDescriptor]:
        """
        Declare the wizard-elicitable scalar inputs to ``__init__``.

        Only the four numeric / seed scalars are returned here. ``objective_scorer``
        and ``scenario_result_id`` are base-scenario lifecycle arguments handled by
        the wizard's standard plumbing; ``context_extractor`` is a ``Callable``
        with a usable default (``global_context``) that the CLI elicitor cannot
        introspect — programmatic callers may override it directly via
        ``build_scenario_from_inputs(..., init_inputs={"context_extractor": ...})``.
        Strategy selection (``scenario_strategies``) lives in
        :meth:`supported_parameters` as an :meth:`initialize_async` ``--kebab-flag``
        argument, not in :meth:`__init__`.

        Returns:
            list[RoleDescriptor]: Four SCALAR roles mirroring the constructor
                defaults.
        """
        return [
            RoleDescriptor(
                name="epsilon",
                description="Exploration probability for the epsilon-greedy selector (0.0 = pure exploit).",
                tag=RoleTag.SCALAR,
                param_type=float,
                default=0.2,
                required=False,
            ),
            RoleDescriptor(
                name="pool_threshold",
                description=(
                    "Minimum per-(context, technique) attempts before the local estimate overrides the pooled rate. "
                    "Set to 1 to disable pooling."
                ),
                tag=RoleTag.SCALAR,
                param_type=int,
                default=3,
                required=False,
            ),
            RoleDescriptor(
                name="max_attempts_per_objective",
                description="Maximum number of techniques to dispatch per objective before giving up.",
                tag=RoleTag.SCALAR,
                param_type=int,
                default=3,
                required=False,
            ),
            RoleDescriptor(
                name="seed",
                description="RNG seed for deterministic technique selection. ``None`` uses a non-deterministic RNG.",
                tag=RoleTag.SCALAR,
                param_type=int,
                default=None,
                required=False,
            ),
        ]

    async def _get_atomic_attacks_async(self) -> list[AtomicAttack]:
        """
        Build one :class:`AdaptiveStep` per objective.

        Each objective gets a freshly constructed step bound to its seed group,
        but all steps share the same selector so learning accumulates across
        objectives. Per-objective, techniques whose ``seed_technique`` is
        incompatible with the seed group are filtered out; objectives left
        with no compatible techniques are skipped.

        The return type is :class:`list[AtomicAttack]` for parity with the base
        ``Scenario._get_atomic_attacks_async`` contract — the orchestrator's
        resume bookkeeping treats steps via the duck-typed attributes
        :class:`AdaptiveStep` provides (``atomic_attack_name``, ``objectives``,
        ``seed_groups``, ``display_group``, ``filter_seed_groups_by_objectives``).
        Execution dispatch in :meth:`_build_execution_graph` calls
        ``step.process_async`` directly, bypassing the default linear policy's
        ``AtomicAttack.run_async`` branch.

        Returns:
            list[AtomicAttack]: One step per objective with at least one
                compatible technique. Empty if every seed group is incompatible
                with every selected technique.

        Raises:
            ValueError: If ``self._objective_target`` is not set, or if
                ``_build_techniques_dict`` finds no usable techniques.
        """
        if self._objective_target is None:
            raise ValueError("objective_target must be set before creating attacks")

        techniques = self._build_techniques_dict(objective_target=self._objective_target)

        selector = AdaptiveTechniqueSelector(
            epsilon=self._epsilon,
            pool_threshold=self._pool_threshold,
            rng=random.Random(self._seed),
        )
        # On resume, replay prior attempt outcomes from persisted metadata.
        self._rehydrate_selector_from_memory(selector=selector, known_techniques=set(techniques))
        # Cache the selector so _build_execution_graph reuses the rehydrated state.
        self._selector = selector

        seed_groups_by_dataset = self._dataset_config.get_seed_attack_groups()
        steps: list[AdaptiveStep] = []
        for dataset_name, seed_groups in seed_groups_by_dataset.items():
            for seed_group in seed_groups:
                step = self._build_step_for_seed_group(
                    dataset_name=dataset_name,
                    seed_group=seed_group,
                    techniques=techniques,
                    selector=selector,
                )
                if step is not None:
                    steps.append(step)

        return cast("list[AtomicAttack]", steps)

    def _build_techniques_dict(
        self,
        *,
        objective_target: PromptTarget,
    ) -> dict[str, TechniqueBundle]:
        """
        Resolve selected strategies into a ``{name: TechniqueBundle}`` map.

        Each bundle carries the inner attack strategy along with the factory's
        ``seed_technique`` and ``adversarial_chat`` so the dispatcher can
        reproduce the static ``AtomicAttack`` execution path per attempt.

        Returns:
            dict[str, TechniqueBundle]: Mapping from technique name to its
                bundle, in the order selected strategies were resolved.

        Raises:
            ValueError: If no techniques remain after filtering. Includes the
                requested techniques and skip reasons.
        """
        selected_techniques = sorted({s.value for s in self._scenario_strategies})
        factories = self._get_attack_technique_factories()
        scoring_config = AttackScoringConfig(objective_scorer=self._objective_scorer)

        techniques: dict[str, TechniqueBundle] = {}
        skipped_no_factory: list[str] = []
        for technique_name in selected_techniques:
            factory = factories.get(technique_name)
            if factory is None:
                skipped_no_factory.append(technique_name)
                logger.warning(f"No factory for technique '{technique_name}', skipping.")
                continue
            technique = factory.create(
                objective_target=objective_target,
                attack_scoring_config=scoring_config,
            )
            techniques[technique_name] = TechniqueBundle(
                attack=technique.attack,
                seed_technique=technique.seed_technique,
                adversarial_chat=factory.adversarial_chat,
            )

        if not techniques:
            suffix = f" (skipped, no factory registered: {sorted(skipped_no_factory)})" if skipped_no_factory else ""
            raise ValueError(
                f"{type(self).__name__}: no usable techniques after resolving strategies. "
                f"Check the --strategies selection.{suffix}"
            )

        return techniques

    def _build_step_for_seed_group(
        self,
        *,
        dataset_name: str,
        seed_group: SeedAttackGroup,
        techniques: dict[str, TechniqueBundle],
        selector: AdaptiveTechniqueSelector,
    ) -> AdaptiveStep | None:
        """
        Build a single :class:`AdaptiveStep` for one ``SeedAttackGroup``.

        Filters the technique pool down to those whose ``seed_technique`` (if
        any) is compatible with this seed group, then constructs an
        :class:`AdaptiveStep` bound to it.

        Returns:
            AdaptiveStep | None: The constructed step, or ``None`` when no
                techniques are compatible (caller skips the objective).

        Raises:
            ValueError: If ``self._objective_target`` is not set (defensive
                guard; ``_get_atomic_attacks_async`` enforces this earlier).
        """
        if self._objective_target is None:  # pragma: no cover - defensive
            raise ValueError("objective_target must be set before creating attacks")

        compatible: dict[str, TechniqueBundle] = {
            name: bundle
            for name, bundle in techniques.items()
            if bundle.seed_technique is None or seed_group.is_compatible_with_technique(technique=bundle.seed_technique)
        }

        if not compatible:
            logger.warning(
                "AdaptiveScenario: no compatible techniques for seed group in dataset '%s' (objective=%r); skipping.",
                dataset_name,
                seed_group.objective.value,
            )
            return None

        adaptive_context = self._context_extractor(seed_group)
        # Derive a deterministic 12-char hash from the objective text so two
        # ``initialize_async`` runs over structurally identical seed groups
        # produce identical step identifiers — a Phase 8g prerequisite for
        # graph-artifact round-trip hash equivalence. ``SeedObjective.id``
        # has a ``uuid.uuid4()`` default-factory (``seed.py:95``) that mints
        # a fresh UUID per in-memory construction, so it cannot be used as
        # a stable resume key here.
        objective_id = hashlib.sha256(seed_group.objective.value.encode("utf-8")).hexdigest()[:12]
        atomic_attack_name = f"{self._atomic_attack_prefix}_{dataset_name}_{objective_id}"

        memory_labels = {
            **self._memory_labels,
            ADAPTIVE_CONTEXT_LABEL: adaptive_context,
        }
        return AdaptiveStep(
            atomic_attack_name=atomic_attack_name,
            display_group=dataset_name,
            objective_target=self._objective_target,
            techniques=compatible,
            selector=selector,
            seed_group=seed_group,
            objective_scorer=self._objective_scorer,
            max_attempts_per_objective=self._max_attempts_per_objective,
            memory_labels=memory_labels,
            adaptive_context=adaptive_context,
        )

    def _build_execution_graph(
        self,
        *,
        steps: Sequence[ScenarioStep] | None = None,
    ) -> StrategyGraph[ScenarioStep, int]:
        """
        Build a linear graph that drives each :class:`AdaptiveStep` via
        ``process_async`` so the ``"success"`` / ``"exhausted"`` outcome
        labels survive into the orchestrator (the default policy from the
        base class would dispatch ``AtomicAttack`` instances through
        ``run_async`` and lose the outcome distinction).

        Args:
            steps: Optional explicit step list. Defaults to
                ``self._atomic_attacks``, mirroring the base class contract.

        Returns:
            StrategyGraph[ScenarioStep, int]: A linear traversal whose actions
                always dispatch via ``step.process_async``.
        """
        effective_steps = list(steps) if steps is not None else list(self._atomic_attacks)
        policy = self._build_adaptive_linear_policy(steps=effective_steps)
        return StrategyGraph(policy=policy)

    def _build_adaptive_linear_policy(
        self,
        *,
        steps: Sequence[ScenarioStep],
    ) -> StrategyPolicy[ScenarioStep, int]:
        """
        Build a linear policy that always dispatches via ``process_async``.

        Each policy action runs ``steps[i].process_async()`` and transitions
        to state ``i + 1``; state ``len(steps)`` is the sole terminal state.
        Unlike :meth:`Scenario._build_default_linear_policy` there's no
        ``isinstance(_step, AtomicAttack)`` branch — adaptive steps always
        go through their own process_async loop so the
        ``"success"``/``"exhausted"`` outcome labels propagate unchanged.

        Args:
            steps: The steps to wrap. Must be non-empty.

        Returns:
            StrategyPolicy[ScenarioStep, int]: A frozen linear policy.

        Raises:
            ValueError: If ``steps`` is empty.
        """
        if not steps:
            raise ValueError("_build_adaptive_linear_policy requires at least one step.")

        terminal_state = len(steps)
        actions: dict[int, PolicyAction[ScenarioStep, int]] = {}

        for index, step in enumerate(steps):

            async def _action(
                graph: StrategyGraph[ScenarioStep, int],
                _step: ScenarioStep = step,
                _next: int = index + 1,
            ) -> tuple[int, ScenarioStepResult | None]:
                graph.bind_current_step(step=_step)
                try:
                    base_result = await _step.process_async()
                    merged_metadata = {"step_name": _step.name, **base_result.metadata}
                    result: ScenarioStepResult | None = ScenarioStepResult(
                        outcome=base_result.outcome,
                        attack_results=list(base_result.attack_results),
                        step_identifier=base_result.step_identifier,
                        metadata=merged_metadata,
                    )
                finally:
                    graph.bind_current_step(step=None)
                return _next, result

            actions[index] = _action

        return StrategyPolicy(
            actions=actions,
            initial_state=0,
            terminal_states=frozenset({terminal_state}),
        )

    def _rehydrate_selector_from_memory(
        self,
        *,
        selector: AdaptiveTechniqueSelector,
        known_techniques: set[str],
    ) -> None:
        """
        Replay persisted dispatch trails into ``selector`` so resume
        preserves learned state.

        Iterates every persisted ``AttackResult`` on the resumed
        ``ScenarioResult`` and calls ``record_outcome`` once per attempt in
        each ``metadata["adaptive_attempts"]`` trail.

        Args:
            selector (AdaptiveTechniqueSelector): A freshly built selector to populate.
            known_techniques (set[str]): Techniques available in the current run.
                Trails referencing unknown techniques (e.g. after a strategies
                change) are skipped so replay can't poison the table.
        """
        if not self._scenario_result_id:
            return

        # Narrow to errors a memory backend would plausibly raise (DB/IO
        # failures, integrity issues). Programmer-level errors propagate.
        try:
            scenario_results = self._memory.get_scenario_results(scenario_result_ids=[self._scenario_result_id])
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning(f"AdaptiveScenario: failed to load prior scenario result for rehydration: {exc}")
            return

        if not scenario_results:
            return

        replayed = 0
        for results_list in scenario_results[0].attack_results.values():
            for result in results_list:
                trail = result.metadata.get("adaptive_attempts") if result.metadata else None
                context = result.metadata.get("adaptive_context") if result.metadata else None
                if not trail or not context:
                    continue
                for step in trail:
                    technique = step.get("technique")
                    outcome = step.get("outcome")
                    if not technique or technique not in known_techniques:
                        continue
                    selector.record_outcome(
                        context=context,
                        technique=technique,
                        success=outcome == "success",
                    )
                    replayed += 1

        if replayed:
            logger.info(f"AdaptiveScenario: rehydrated selector with {replayed} prior attempt(s).")
