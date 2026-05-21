# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``ScenarioPipeline`` — sequential composition of inner scenarios as phases.

Direct response to the review note that the simple cross-scenario composition
case ("run scenario A, then scenario B, then scenario C") should not require
authoring a custom branching policy. ``ScenarioPipeline`` is a thin
``Scenario`` subclass that walks a list of :class:`PhaseSpec` instances in
order, dispatching each phase's inner scenario through its own
``initialize_async`` / ``run_async`` lifecycle. The pipeline itself owns a
linear :class:`StrategyGraph` over integer states so it stays consistent with
the rest of the scenario core, but callers never need to touch the graph
vocabulary — they hand the pipeline a list of phase specs.

Phase-level conditional skipping is expressed via :class:`ConditionalPhaseSpec`,
whose ``skip_when`` predicate receives a read-only :class:`PipelineContext`
snapshot of prior phase executions.

Pipeline-level resume is intentionally out of scope for the initial release:
each pipeline run constructs fresh inner scenarios with ``scenario_result_id``
unset. Pipelines wanting to resume can supply phase factories that recreate
inner scenarios with previously-persisted scenario result ids.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Optional, cast

from pyrit.common import apply_defaults
from pyrit.identifiers import ComponentIdentifier
from pyrit.models import Score
from pyrit.scenario.core.dataset_configuration import DatasetConfiguration
from pyrit.scenario.core.input_schema import RoleDescriptor, RoleTag
from pyrit.scenario.core.scenario import BaselineAttackPolicy, Scenario
from pyrit.scenario.core.scenario_step import ScenarioStep, ScenarioStepResult
from pyrit.scenario.core.scenario_strategy import ScenarioStrategy
from pyrit.scenario.core.strategy_graph import (
    PolicyAction,
    StrategyGraph,
    StrategyPolicy,
)
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable, Mapping, Sequence

    from pyrit.models import MessagePiece
    from pyrit.models.scenario_result import ScenarioResult
    from pyrit.models.score import ScoreType
    from pyrit.scenario.core.atomic_attack import AtomicAttack
    from pyrit.scenario.core.attack_technique_factory import AttackTechniqueFactory
    from pyrit.score.scorer import Scorer

logger = logging.getLogger(__name__)


_PHASE_OUTCOME_COMPLETED = "completed"
_PHASE_OUTCOME_SKIPPED = "skipped"


@dataclass(frozen=True)
class PhaseExecution:
    """
    Snapshot of one phase's execution result.

    Surfaced to :class:`PipelineContext` for :class:`ConditionalPhaseSpec`
    predicates and accessible after the pipeline completes via
    :attr:`ScenarioPipeline.phase_executions`.

    Attributes:
        name (str): The phase's name as declared in :class:`PhaseSpec`.
        outcome (str): Either ``"completed"`` (phase ran to completion) or
            ``"skipped"`` (a :class:`ConditionalPhaseSpec` predicate elided
            execution). Failures propagate as exceptions out of
            ``ScenarioPipeline.run_async`` rather than landing here.
        scenario_result (ScenarioResult | None): The phase's inner
            :class:`ScenarioResult` if the phase ran; ``None`` if the phase
            was skipped.
    """

    name: str
    outcome: str
    scenario_result: Optional[ScenarioResult] = None


@dataclass(frozen=True)
class PipelineContext:
    """
    Read-only view of prior phase executions, supplied to predicates.

    :class:`ConditionalPhaseSpec.skip_when` receives one of these so predicates
    can branch on the outcomes (or full ``ScenarioResult`` payloads) of phases
    that already ran. Tuples (not lists) make accidental mutation by the
    predicate impossible.

    Attributes:
        completed_phase_names (tuple[str, ...]): Names of every phase that has
            either completed or been skipped so far, in dispatch order.
        completed_phase_outcomes (tuple[str, ...]): The matching outcomes
            (``"completed"`` or ``"skipped"``) in the same order.
        phase_executions (tuple[PhaseExecution, ...]): Full execution snapshots
            including the inner ``ScenarioResult`` for non-skipped phases.
    """

    completed_phase_names: tuple[str, ...] = ()
    completed_phase_outcomes: tuple[str, ...] = ()
    phase_executions: tuple[PhaseExecution, ...] = ()


@dataclass(frozen=True)
class PhaseSpec:
    """
    Declarative description of one pipeline phase.

    A phase wraps a single inner ``Scenario`` that is constructed fresh for
    each pipeline run via ``scenario_factory``. The pipeline owns
    ``initialize_async`` and ``run_async`` for the inner scenario, merging
    ``init_async_kwargs`` with auto-injected values (``objective_target`` and
    ``memory_labels`` flow from the outer pipeline unless explicitly
    overridden in ``init_async_kwargs``).

    Attributes:
        name (str): A unique, human-readable label for this phase. Used as
            the bucket key in :class:`ScenarioResult.attack_results` so
            downstream consumers can pull per-phase results from the
            pipeline-level result.
        scenario_factory (Callable[[], Scenario]): Zero-arg callable that
            builds a fresh, uninitialized inner scenario instance. Called
            once per pipeline run when the phase's turn arrives. Use
            ``lambda: MyScenario(target=..., scorer=...)`` for the common
            case where construction args don't depend on prior phases.
        init_async_kwargs (Mapping[str, Any]): Kwargs forwarded verbatim
            to the inner scenario's ``initialize_async``. ``objective_target``
            and ``memory_labels`` are auto-injected from the outer pipeline
            unless explicitly provided here.
    """

    name: str
    scenario_factory: Callable[[], Scenario]
    init_async_kwargs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class ConditionalPhaseSpec(PhaseSpec):
    """
    Phase that runs only when ``skip_when`` returns False for the current context.

    The predicate is evaluated once at dispatch time, after every prior phase
    has either completed or been skipped. Predicates that need access to a
    prior phase's ``ScenarioResult`` (e.g. to inspect successful attacks)
    should look it up by name from :attr:`PipelineContext.phase_executions`.

    Attributes:
        skip_when (Callable[[PipelineContext], bool]): Returns True to skip
            this phase, False to run it. Defaults to never skipping (which is
            equivalent to using a plain :class:`PhaseSpec`).
    """

    skip_when: Callable[[PipelineContext], bool] = field(default=lambda _ctx: False)


class ScenarioPipelineStrategy(ScenarioStrategy):
    """
    Single-member strategy enum for :class:`ScenarioPipeline`.

    Pipelines don't expose a technique selection menu — composition is
    expressed via the ``phases`` constructor argument. The strategy enum
    exists only to satisfy the base :class:`Scenario` contract.
    """

    DEFAULT = ("default", {"all"})

    @classmethod
    def get_aggregate_tags(cls) -> set[str]:
        """Return the strategy aggregate tags this enum exposes."""
        return {"all"}


class _PipelineNoopScorer(TrueFalseScorer):
    """
    Stand-in scorer for :class:`ScenarioPipeline` when no pipeline-level scorer is provided.

    The pipeline doesn't directly score anything — each inner scenario brings
    its own scorer. But ``Scenario.__init__`` requires a ``Scorer`` for its
    identifier graph, so we hand it a deterministic no-op that always emits a
    false score. Never actually invoked during pipeline execution because the
    pipeline does not call its base ``objective_scorer`` on the phase results.
    """

    def __init__(self) -> None:
        super().__init__(validator=ScorerPromptValidator())

    async def _score_piece_async(
        self,
        message_piece: MessagePiece,
        *,
        objective: Optional[str] = None,
    ) -> list[Score]:
        # ``MessagePiece.id`` is typed Optional but is populated for every persisted
        # piece (its default factory mints a UUID). The cast satisfies ``Score.__init__``
        # without paying a runtime ``assert`` that would never fire in practice.
        return [
            Score(
                score_value="False",
                score_value_description="ScenarioPipeline no-op scorer; not used during pipeline execution.",
                score_type=cast("ScoreType", "true_false"),
                score_rationale="Pipeline does not score attack results directly; inner scenarios own scoring.",
                message_piece_id=cast("str | uuid.UUID", message_piece.id),
                scorer_class_identifier=self.get_identifier(),
                objective=objective,
            )
        ]

    def _build_identifier(self) -> ComponentIdentifier:
        return ComponentIdentifier.of(self)


class _ScenarioPipelinePhaseStep(ScenarioStep):
    """
    One pipeline phase, dispatched by :class:`ScenarioPipeline`'s execution graph.

    Holds a :class:`PhaseSpec` and a back-reference to the parent pipeline
    (so the predicate can read prior-phase context and the inner scenario can
    inherit the pipeline's ``objective_target`` and ``memory_labels``).
    Duck-types the :class:`AtomicAttack` attributes the base ``Scenario``
    orchestrator inspects (``atomic_attack_name``, ``display_group``,
    ``seed_groups``, ``objectives``, ``technique_eval_hash``,
    ``filter_seed_groups_by_objectives``, ``drop_seed_groups_with_hashes``)
    so the pipeline can plug into the existing orchestrator without
    special-casing branching scenarios.

    The step's ``process_async`` evaluates the conditional predicate (if any),
    builds the inner scenario via the factory, initializes it with merged
    kwargs, and runs it. The returned :class:`ScenarioStepResult` carries an
    empty ``attack_results`` list — inner scenarios persist their own results
    against their own ``scenario_result_id``, so re-emitting them here would
    duplicate-persist them under the pipeline's own ``scenario_result_id``.
    """

    _OUTPUTS: ClassVar[tuple[str, ...]] = (_PHASE_OUTCOME_COMPLETED, _PHASE_OUTCOME_SKIPPED)

    #: Marker for ``graph_artifact.build_graph_artifact`` (Phase 8g): each phase
    #: holds a ``Callable`` scenario factory whose closure cannot be reconstructed
    #: from primitive args. Encoding via ``ComponentIdentifier.to_dict()`` is the
    #: only sound round-trip path.
    GRAPH_ARTIFACT_OPAQUE: ClassVar[bool] = True

    def __init__(
        self,
        *,
        spec: PhaseSpec,
        index: int,
        pipeline: ScenarioPipeline,
    ) -> None:
        """
        Initialize the phase step.

        Args:
            spec (PhaseSpec): The phase declaration.
            index (int): Zero-based position in the pipeline's phase list.
                Surfaced into step metadata for diagnostics and used as the
                policy state key in the pipeline's linear graph.
            pipeline (ScenarioPipeline): Back-reference used to (a) inherit
                ``objective_target`` and ``memory_labels`` into the inner
                scenario, and (b) feed :class:`PipelineContext` to
                :class:`ConditionalPhaseSpec` predicates.
        """
        self.name = spec.name
        self.outputs = list(self._OUTPUTS)
        self._spec = spec
        self._index = index
        self._pipeline = pipeline

        # Duck-typed AtomicAttack-like attributes so the orchestrator's
        # display-group map and resume bookkeeping continue to work without
        # special-casing pipeline phases.
        self.atomic_attack_name = spec.name
        self.display_group = spec.name
        self.seed_groups: list[Any] = []
        self.objectives: list[str] = []

        # ``Scenario._get_completed_objective_hashes_for_attack`` reads
        # ``technique_eval_hash`` to scope its rows-by-attribution lookup.
        # Pipeline phases have no per-row attribution at the pipeline level
        # (inner scenarios persist under their own ids), so the value is a
        # stable sentinel that can never collide with a real eval hash.
        self.technique_eval_hash: str = f"pipeline-phase::{spec.name}"

    def filter_seed_groups_by_objectives(self, *, remaining_objectives: list[str]) -> None:
        """No-op: pipeline phases own no seed groups at the pipeline level."""

    def drop_seed_groups_with_hashes(self, *, hashes: set[str]) -> None:
        """No-op: pipeline phases own no seed groups at the pipeline level."""

    async def process_async(self) -> ScenarioStepResult:
        """
        Run the inner scenario (or skip via ``ConditionalPhaseSpec`` predicate).

        Returns:
            ScenarioStepResult: Outcome is ``"completed"`` if the inner
                scenario ran, ``"skipped"`` if the predicate elided the
                phase. ``attack_results`` is always empty — inner scenarios
                own their own persistence. ``metadata`` carries
                ``phase_name`` / ``phase_index`` / (when run)
                ``inner_scenario_result_id`` for diagnostics and downstream
                lookup.

        Raises:
            TypeError: If ``scenario_factory`` returns an object that is not
                a :class:`Scenario` instance.
        """
        spec = self._spec
        pipeline = self._pipeline

        if isinstance(spec, ConditionalPhaseSpec):
            context = pipeline._snapshot_pipeline_context()
            if spec.skip_when(context):
                logger.info(
                    "ScenarioPipeline '%s' phase %d/%d '%s': SKIPPED by predicate",
                    pipeline._name,
                    self._index + 1,
                    len(pipeline._phases),
                    spec.name,
                )
                execution = PhaseExecution(name=spec.name, outcome=_PHASE_OUTCOME_SKIPPED, scenario_result=None)
                pipeline._record_phase_execution(execution=execution)
                return ScenarioStepResult(
                    outcome=_PHASE_OUTCOME_SKIPPED,
                    attack_results=[],
                    metadata={
                        "phase_name": spec.name,
                        "phase_index": self._index,
                        "skipped": True,
                    },
                )

        logger.info(
            "ScenarioPipeline '%s' phase %d/%d '%s': running",
            pipeline._name,
            self._index + 1,
            len(pipeline._phases),
            spec.name,
        )

        inner_scenario = spec.scenario_factory()
        if not isinstance(inner_scenario, Scenario):
            raise TypeError(
                f"PhaseSpec(name={spec.name!r}).scenario_factory() returned "
                f"{type(inner_scenario).__name__!r}, expected a Scenario instance."
            )

        init_kwargs = dict(spec.init_async_kwargs)
        if "objective_target" not in init_kwargs and pipeline._objective_target is not None:
            init_kwargs["objective_target"] = pipeline._objective_target
        if "memory_labels" not in init_kwargs and pipeline._memory_labels:
            init_kwargs["memory_labels"] = dict(pipeline._memory_labels)

        await inner_scenario.initialize_async(**init_kwargs)
        inner_result = await inner_scenario.run_async()

        execution = PhaseExecution(name=spec.name, outcome=_PHASE_OUTCOME_COMPLETED, scenario_result=inner_result)
        pipeline._record_phase_execution(execution=execution)

        return ScenarioStepResult(
            outcome=_PHASE_OUTCOME_COMPLETED,
            attack_results=[],
            metadata={
                "phase_name": spec.name,
                "phase_index": self._index,
                "inner_scenario_result_id": str(inner_result.id) if inner_result.id is not None else None,
            },
        )

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build the behavioral identity for this phase step.

        The factory callable is opaque — only the phase name and index are
        captured. ``GRAPH_ARTIFACT_OPAQUE = True`` tells artifact serializers
        to round-trip via the identifier hash rather than introspecting the
        constructor closure.

        Returns:
            ComponentIdentifier: The frozen identity snapshot.
        """
        return ComponentIdentifier.of(
            self,
            params={
                "phase_name": self.name,
                "phase_index": self._index,
                "outputs": list(self.outputs),
            },
        )


class ScenarioPipeline(Scenario):
    """
    Compose multiple scenarios as sequential phases.

    The pipeline runs each phase's inner scenario in declaration order. Each
    inner scenario owns its own dataset, strategies, scorer, and
    ``ScenarioResult`` — the pipeline's own ``ScenarioResult`` records the
    composition and per-phase outcomes, not the inner ``AttackResult``s
    themselves. To inspect per-phase results after a pipeline run, walk
    :attr:`phase_executions` and pull ``execution.scenario_result.attack_results``
    for each completed phase.

    Example::

        pipeline = ScenarioPipeline(
            phases=[
                PhaseSpec(name="sweep", scenario_factory=lambda: RapidResponse(...)),
                ConditionalPhaseSpec(
                    name="deep_dive",
                    scenario_factory=lambda: RapidResponse(...),
                    init_async_kwargs={"scenario_strategies": [RapidResponseStrategy.ManyShot]},
                    skip_when=lambda ctx: "sweep" not in ctx.completed_phase_names,
                ),
            ],
        )
        await pipeline.initialize_async(objective_target=target)
        result = await pipeline.run_async()
        for execution in pipeline.phase_executions:
            print(execution.name, execution.outcome)

    Pipeline-level resume is not supported in this release: pass
    ``scenario_result_id=None`` (the default) for every run. Inner scenarios
    that need their own resume should manage that themselves inside their
    ``scenario_factory``.
    """

    VERSION: ClassVar[int] = 1

    #: Pipelines compose other scenarios; baseline injection only makes sense
    #: per inner scenario, so prevent the base class from prepending one at
    #: the pipeline level. Inner scenarios independently respect their own
    #: ``BASELINE_ATTACK_POLICY``.
    BASELINE_ATTACK_POLICY: ClassVar[BaselineAttackPolicy] = BaselineAttackPolicy.Forbidden

    @apply_defaults
    def __init__(
        self,
        *,
        phases: Sequence[PhaseSpec],
        objective_scorer: Optional[Scorer] = None,
        name: Optional[str] = None,
        scenario_result_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the pipeline.

        Args:
            phases (Sequence[PhaseSpec]): Ordered list of phase declarations.
                Must be non-empty. Phase names must be unique across the list
                so the per-phase result bucket can be addressed unambiguously.
            objective_scorer (Optional[Scorer]): Pipeline-level scorer. Not
                used during execution (each inner scenario brings its own
                scorer) but recorded on the pipeline's
                :class:`ScenarioResult` for downstream introspection. A
                deterministic no-op scorer is used when not provided.
            name (Optional[str]): Display name for the pipeline. Defaults to
                ``"ScenarioPipeline"``.
            scenario_result_id (Optional[str]): Must be ``None`` in this
                release — pipeline-level resume is not yet supported.

        Raises:
            ValueError: If ``phases`` is empty, contains duplicate names, or
                ``scenario_result_id`` is supplied.
        """
        if not phases:
            raise ValueError("ScenarioPipeline requires at least one PhaseSpec.")

        names = [phase.name for phase in phases]
        seen: set[str] = set()
        duplicates: list[str] = []
        for phase_name in names:
            if phase_name in seen and phase_name not in duplicates:
                duplicates.append(phase_name)
            seen.add(phase_name)
        if duplicates:
            raise ValueError(
                f"PhaseSpec names must be unique within a ScenarioPipeline; duplicates: {sorted(duplicates)!r}"
            )

        if scenario_result_id is not None:
            raise ValueError(
                "ScenarioPipeline does not support pipeline-level resume yet. "
                "Pass scenario_result_id=None and recreate inner scenarios via PhaseSpec.scenario_factory."
            )

        self._phases: list[PhaseSpec] = list(phases)
        self._phase_executions: list[PhaseExecution] = []

        effective_scorer = objective_scorer if objective_scorer is not None else _PipelineNoopScorer()

        super().__init__(
            name=name or type(self).__name__,
            version=self.VERSION,
            strategy_class=ScenarioPipelineStrategy,
            objective_scorer=effective_scorer,
        )

    @classmethod
    def get_strategy_class(cls) -> type[ScenarioStrategy]:
        """Return the (single-member) strategy enum class."""
        return ScenarioPipelineStrategy

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        """Return the only strategy member."""
        return ScenarioPipelineStrategy.DEFAULT

    @classmethod
    def default_dataset_config(cls) -> DatasetConfiguration:
        """
        Return an empty dataset configuration.

        Each phase's inner scenario brings its own dataset, so the pipeline
        has nothing to resolve at the outer level.

        Returns:
            DatasetConfiguration: An empty configuration used by the base
                ``initialize_async`` when the caller does not supply one.
        """
        return DatasetConfiguration()

    @classmethod
    def input_schema(cls) -> list[RoleDescriptor]:
        """
        Declare the rich-object input the wizard / artifact must capture.

        ``phases`` is opaque because each :class:`PhaseSpec` holds a callable
        factory whose closure cannot be reconstructed from primitive args.
        Pipelines must be authored programmatically — round-tripping a
        pipeline via a saved graph artifact is not supported in v1, because
        neither the ``Callable`` factory nor ``PhaseSpec.init_async_kwargs``
        (a :class:`types.MappingProxyType`) is YAML-serializable. See
        :class:`tests.unit.scenario.composite.test_scenario_pipeline.TestArtifactRoundTripNotSupported`
        for the regression pin.

        Returns:
            list[RoleDescriptor]: Two roles — ``phases`` (opaque, required)
                and ``name`` (scalar, optional).
        """
        return [
            RoleDescriptor(
                name="phases",
                description=(
                    "Ordered list of PhaseSpec / ConditionalPhaseSpec instances declaring the inner scenarios "
                    "to run as pipeline phases. Each holds a callable factory and is therefore opaque."
                ),
                tag=RoleTag.OPAQUE,
                required=True,
            ),
            RoleDescriptor(
                name="name",
                description="Display name for the pipeline. Defaults to the class name.",
                tag=RoleTag.SCALAR,
                param_type=str,
                default=None,
                required=False,
            ),
        ]

    @property
    def phases(self) -> tuple[PhaseSpec, ...]:
        """Return the declared phase specs as an immutable tuple."""
        return tuple(self._phases)

    @property
    def phase_executions(self) -> tuple[PhaseExecution, ...]:
        """Return snapshots of phase executions from the most recent run."""
        return tuple(self._phase_executions)

    def _get_attack_technique_factories(self) -> dict[str, AttackTechniqueFactory]:
        """
        Return an empty factory map: pipelines do not own techniques directly.

        Each inner scenario owns its own factories; the pipeline-level
        registry inspection (used by ``policy_to_spec``) is intentionally
        empty so introspection stays side-effect-free at the pipeline layer.

        Returns:
            dict[str, AttackTechniqueFactory]: Always empty.
        """
        return {}

    async def _get_atomic_attacks_async(self) -> list[AtomicAttack]:
        """
        Materialize one :class:`_ScenarioPipelinePhaseStep` per phase.

        The orchestrator's resume bookkeeping reads ``atomic_attack_name``,
        ``display_group``, ``seed_groups``, ``objectives``, and
        ``technique_eval_hash`` from each returned step. The phase step
        duck-types all of these so the base class can plug pipelines into
        the existing execution flow without branching.

        Returns:
            list[AtomicAttack]: One step per phase, cast to satisfy the
                base contract. Phase steps subclass :class:`ScenarioStep`,
                not :class:`AtomicAttack`; dispatch happens through
                ``process_async`` via the pipeline's custom execution graph
                rather than through ``AtomicAttack.run_async``.
        """
        steps: list[ScenarioStep] = [
            _ScenarioPipelinePhaseStep(spec=spec, index=index, pipeline=self) for index, spec in enumerate(self._phases)
        ]
        return cast("list[AtomicAttack]", steps)

    async def _get_remaining_atomic_attacks_async(self) -> list[AtomicAttack]:
        """
        Skip the orchestrator's atomic-attack resume bookkeeping for pipelines.

        Pipeline-level resume is not supported in this release (every run
        constructs fresh inner scenarios with no scenario_result_id), so
        ``_get_completed_objective_hashes_for_attack`` would return an
        empty set for every phase step anyway. Returning ``self._atomic_attacks``
        directly avoids invoking the AtomicAttack-shaped helpers
        (``technique_eval_hash`` lookup, ``drop_seed_groups_with_hashes``)
        on the pipeline phase steps even though they expose stable stubs
        for those attributes.

        Returns:
            list[AtomicAttack]: All phase steps, in declaration order.
        """
        return self._atomic_attacks

    def _build_execution_graph(
        self,
        *,
        steps: Optional[Sequence[ScenarioStep]] = None,
    ) -> StrategyGraph[ScenarioStep, int]:
        """
        Build a linear integer-state graph that walks the supplied phase steps.

        Each phase becomes one policy action keyed by its zero-based index.
        State ``len(phases)`` is the terminal state. Reset
        ``self._phase_executions`` on every invocation so each pipeline run
        starts with a clean execution log for :class:`PipelineContext`.

        Args:
            steps: Phase steps to dispatch. The base ``_execute_scenario_async``
                passes the resume-filtered ``_atomic_attacks`` list here.

        Returns:
            StrategyGraph[ScenarioStep, int]: A linear graph with one state
                per phase, terminating after the final phase's action runs.

        Raises:
            ValueError: If the resolved step list is empty.
        """
        if steps is None:
            steps = self._atomic_attacks

        phase_steps = list(steps)
        if not phase_steps:
            raise ValueError("ScenarioPipeline cannot build an execution graph without phase steps.")

        self._phase_executions = []

        actions: dict[int, PolicyAction[ScenarioStep, int]] = {}
        for index, step in enumerate(phase_steps):
            actions[index] = self._build_phase_action(index=index, step=step)

        terminal_state = len(phase_steps)
        return StrategyGraph(
            policy=StrategyPolicy(
                actions=actions,
                initial_state=0,
                terminal_states=frozenset({terminal_state}),
            ),
        )

    def _build_phase_action(
        self,
        *,
        index: int,
        step: ScenarioStep,
    ) -> PolicyAction[ScenarioStep, int]:
        """
        Build the policy action for one phase.

        The action binds the phase step as the current step (so the
        orchestrator can resolve ``graph.current_step`` mid-dispatch),
        invokes ``step.process_async``, and advances to the next integer
        state. The step's metadata (``phase_name``, ``phase_index``) is
        merged into the result's metadata so the orchestrator's logging /
        step_identifier path sees a consistent step name.

        Args:
            index: Zero-based phase index in the linear graph.
            step: The phase step to dispatch from this state.

        Returns:
            PolicyAction: An async action that runs the phase and transitions
                to state ``index + 1``.
        """
        next_state = index + 1

        async def _phase_action(
            graph: StrategyGraph[ScenarioStep, int],
        ) -> tuple[int, ScenarioStepResult | None]:
            graph.bind_current_step(step=step)
            try:
                base_result = await step.process_async()
                merged_metadata = {
                    "step_name": step.name,
                    "phase_index": index,
                    **base_result.metadata,
                }
                result = ScenarioStepResult(
                    outcome=base_result.outcome,
                    attack_results=list(base_result.attack_results),
                    step_identifier=base_result.step_identifier,
                    metadata=merged_metadata,
                )
            finally:
                graph.bind_current_step(step=None)
            return next_state, result

        return _phase_action

    def _record_phase_execution(self, *, execution: PhaseExecution) -> None:
        """Append a phase execution snapshot to the pipeline's run-scoped log."""
        self._phase_executions.append(execution)

    def _snapshot_pipeline_context(self) -> PipelineContext:
        """
        Build the immutable :class:`PipelineContext` view for predicate evaluation.

        Returns:
            PipelineContext: A frozen snapshot of every phase execution that
                has completed (or been skipped) so far during this pipeline
                run.
        """
        executions = tuple(self._phase_executions)
        return PipelineContext(
            completed_phase_names=tuple(execution.name for execution in executions),
            completed_phase_outcomes=tuple(execution.outcome for execution in executions),
            phase_executions=executions,
        )
