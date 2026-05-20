# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``BroadSweepThenDeepDive`` — Phase 7 validation artifact for branching scenarios.

First real consumer of the non-linear ``StrategyGraph`` policy. The flow is:

1. **Sweep phase**: run one fast single-turn ``AtomicAttack`` across every
   provided seed group. For each ``AttackResult`` produced, the configured
   ``OutcomeScorer`` classifies the model's final response. Categories
   (``display_group``) that emit the configured ``weakness_label`` are
   tracked.
2. **Branch**:
   - If at least one category was flagged, transition to **Deep dive**.
   - Otherwise terminate immediately.
3. **Deep dive phase**: run each provided multi-turn ``AtomicAttack`` ONLY
   against the categories the sweep flagged. Untargeted categories are
   skipped; their names are stamped into ``ScenarioStepResult.metadata``
   for diagnostics.

This file intentionally bundles its two custom ``ScenarioStep`` subclasses
(``CategoryAggregatingSweepStep``, ``FilteredDeepDiveStep``) inline rather
than promoting them to ``pyrit/scenario/core/``. Phase 9 will extract any
of them once a second scenario shows the same shape.

Scope: this scenario takes its sweep + deep-dive ``AtomicAttack`` lists
explicitly via the constructor (no registry-driven technique selection).
The goal of Phase 7 is to prove the branching policy works end-to-end
without dragging in registry/factory machinery; subclasses with real
technique selection can override ``_build_atomic_attacks_for_phases``.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Optional, cast

from pyrit.common import apply_defaults
from pyrit.models import Message
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
from pyrit.score.decorators.outcome_scorer import OutcomeScorer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pyrit.identifiers import ComponentIdentifier
    from pyrit.models import AttackResult
    from pyrit.scenario.core.atomic_attack import AtomicAttack
    from pyrit.scenario.core.attack_technique_factory import AttackTechniqueFactory
    from pyrit.score import TrueFalseScorer

logger = logging.getLogger(__name__)


class SweepThenDeepDiveState(str, Enum):
    """
    The three states that drive ``BroadSweepThenDeepDive``.

    Inherits ``str`` for the canonical ``(str, Enum)`` design language used
    by ``ScenarioCoreState``, ``BaselineAttackPolicy``, and ``AttackOutcome``: the
    enum members serialize naturally in identifiers and logs.

    The two terminal states are distinct so that downstream consumers can
    tell why the scenario stopped (``ALL_SAFE`` = nothing to escalate;
    ``COMPLETE`` = deep dive finished).
    """

    SWEEPING = "sweeping"
    DEEP_DIVING = "deep_diving"
    COMPLETE = "complete"
    ALL_SAFE = "all_safe"


class BroadSweepThenDeepDiveStrategy(ScenarioStrategy):
    """
    Single-member strategy enum for ``BroadSweepThenDeepDive``.

    This scenario doesn't run a technique-selection menu — the sweep and
    deep-dive attacks are constructor inputs. The strategy enum exists only
    to satisfy the base ``Scenario`` contract; a future migration to
    registry-driven techniques would replace this with a dynamic
    ``build_strategy_class_from_specs`` call.
    """

    DEFAULT = ("default", {"all"})

    @classmethod
    def get_aggregate_tags(cls) -> set[str]:
        """Return the strategy aggregate tags this enum exposes."""
        return {"all"}


class CategoryAggregatingSweepStep(ScenarioStep):
    """
    Run one ``AtomicAttack`` across all seed groups and classify each result.

    The wrapped attack is invoked exactly once via ``AtomicAttack.run_async``.
    Each ``AttackResult`` produced is then handed to the configured
    ``OutcomeScorer.resolve_outcome_async``; the result's ``display_group``
    serves as the category key. If any category's label matches
    ``weakness_label``, the step emits ``"found_weaknesses"``; otherwise it
    emits ``"all_safe"``. The set of weak category names is stamped into
    ``metadata['weak_categories']`` for the downstream deep-dive step.

    Duck-types AtomicAttack-like attributes (``atomic_attack_name``,
    ``display_group``, ``objectives``, ``seed_groups``,
    ``filter_seed_groups_by_objectives``) so the orchestrator's existing
    resume bookkeeping does not need to special-case branching scenarios.
    """

    _OUTPUTS: ClassVar[tuple[str, ...]] = ("found_weaknesses", "all_safe")

    #: Marker for ``graph_artifact.build_graph_artifact`` (Phase 8g): this step's
    #: constructor takes a bound ``OutcomeScorer`` and an ``AtomicAttack`` whose
    #: factory closures cannot be re-derived from primitive args. Encoding via
    #: ``ComponentIdentifier.to_dict()`` is the only sound round-trip path.
    GRAPH_ARTIFACT_OPAQUE: ClassVar[bool] = True

    def __init__(
        self,
        *,
        atomic_attack_name: str,
        atomic_attack: AtomicAttack,
        outcome_scorer: OutcomeScorer,
        weakness_label: str,
        max_concurrency: int = 1,
    ) -> None:
        """
        Initialize the sweep step.

        Args:
            atomic_attack_name (str): Step name used for the
                ``ScenarioStepResult.metadata['step_name']`` stamp and for
                the orchestrator's resume-by-name path.
            atomic_attack (AtomicAttack): The single-turn attack to dispatch.
                Its ``seed_groups``, ``display_group``, and ``objectives``
                are surfaced upward so the orchestrator sees a familiar
                shape.
            outcome_scorer (OutcomeScorer): Decorator that turns the wrapped
                scorer's output into a transition label. Must declare
                ``weakness_label`` as one of its outputs.
            weakness_label (str): The label emitted by ``outcome_scorer``
                that indicates a category was breached and should be
                escalated to deep dive.
            max_concurrency (int): Forwarded to ``AtomicAttack.run_async``.

        Raises:
            ValueError: If ``weakness_label`` is not declared in
                ``outcome_scorer.outcomes``.
        """
        if weakness_label not in outcome_scorer.outcomes:
            raise ValueError(
                f"weakness_label {weakness_label!r} is not declared as an outcome of the "
                f"supplied OutcomeScorer (declared: {outcome_scorer.outcomes!r})."
            )

        self.name = atomic_attack_name
        self.outputs = list(self._OUTPUTS)
        self._atomic = atomic_attack
        self._outcome_scorer = outcome_scorer
        self._weakness_label = weakness_label
        self._max_concurrency = max_concurrency

        # Duck-typed AtomicAttack-like attributes so the orchestrator's
        # resume bookkeeping continues to work without changes.
        self.atomic_attack_name = atomic_attack_name
        self.display_group = atomic_attack.display_group
        self.seed_groups = list(atomic_attack.seed_groups)
        self.objectives = list(atomic_attack.objectives)

    def filter_seed_groups_by_objectives(self, *, remaining_objectives: list[str]) -> None:
        """
        Defer to the wrapped attack's seed-group filter.

        Mirrors the ``AtomicAttack`` shape so ``Scenario._execute_scenario_async``
        can prune already-completed objectives without special-casing this
        step type.

        Args:
            remaining_objectives: Remaining objective strings the orchestrator
                wants this step to cover.
        """
        self._atomic.filter_seed_groups_by_objectives(remaining_objectives=remaining_objectives)
        self.seed_groups = list(self._atomic.seed_groups)
        self.objectives = list(self._atomic.objectives)

    async def process_async(self) -> ScenarioStepResult:
        """
        Execute the sweep and classify each result.

        Returns:
            ScenarioStepResult: Outcome is ``"found_weaknesses"`` if any
                category emitted ``self._weakness_label``, else
                ``"all_safe"``. ``metadata['weak_categories']`` carries the
                set of flagged categories and ``metadata['category_outcomes']``
                carries the full per-category label mapping for diagnostics.
        """
        executor_result = await self._atomic.run_async(
            max_concurrency=self._max_concurrency,
            return_partial_on_failure=True,
        )
        attack_results: list[AttackResult] = list(executor_result.completed_results)

        weak_categories: set[str] = set()
        category_outcomes: dict[str, str] = {}

        for result in attack_results:
            category = self._category_key(result)
            label = await self._classify_async(result=result)
            category_outcomes[category] = label
            if label == self._weakness_label:
                weak_categories.add(category)

        outcome = "found_weaknesses" if weak_categories else "all_safe"
        return ScenarioStepResult(
            outcome=outcome,
            attack_results=attack_results,
            metadata={
                "weak_categories": weak_categories,
                "category_outcomes": category_outcomes,
            },
        )

    async def _classify_async(self, *, result: AttackResult) -> str:
        """
        Map a single ``AttackResult`` to one of the scorer's labels.

        Returns the ``UNSCORED`` sentinel when the result has no
        ``last_response`` to feed the scorer — the surrounding policy can
        choose to treat that as a weakness or not.

        Args:
            result: The attack result whose final response should be classified.

        Returns:
            str: The label emitted by ``self._outcome_scorer``.
        """
        if result.last_response is None:
            return OutcomeScorer.UNSCORED
        message = Message(message_pieces=[result.last_response], skip_validation=True)
        return await self._outcome_scorer.resolve_outcome_async(message, objective=result.objective)

    def _category_key(self, result: AttackResult) -> str:
        """Return the display-group key used to bucket this result by category."""
        return self.display_group or ""

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build the behavioral identity for this sweep step.

        Nests the wrapped attack's identifier under ``children["sweep_attack"]``
        and the outcome scorer under ``children["outcome_scorer"]`` so
        hash drift in either propagates upward.

        Returns:
            ComponentIdentifier: The frozen identity snapshot.
        """
        from pyrit.identifiers import ComponentIdentifier

        return ComponentIdentifier.of(
            self,
            params={
                "atomic_attack_name": self.atomic_attack_name,
                "outputs": list(self.outputs),
                "weakness_label": self._weakness_label,
            },
            children={
                "sweep_attack": self._atomic.get_identifier(),
                "outcome_scorer": self._outcome_scorer.wrapped_scorer.get_identifier(),
            },
        )


class FilteredDeepDiveStep(ScenarioStep):
    """
    Run each supplied ``AtomicAttack`` only if its category was flagged weak.

    Receives a ``weak_categories_ref`` callable that returns the live set
    of weak categories (typically a closure over the sweep step's result).
    For each wrapped atomic attack, ``display_group`` is checked against
    the live set; non-matching atomics are skipped. Always emits the
    single ``"done"`` outcome — deep-dive results are aggregated regardless
    of their individual ``AttackOutcome``.

    Duck-types AtomicAttack-like attributes (``atomic_attack_name``,
    ``display_group``, ``objectives``, ``seed_groups``) for orchestrator
    bookkeeping.
    """

    _OUTPUTS: ClassVar[tuple[str, ...]] = ("done",)

    #: Marker for ``graph_artifact.build_graph_artifact`` (Phase 8g): this step's
    #: constructor takes a ``Callable`` closure (``weak_categories_ref``) that
    #: cannot be re-derived from primitive args. Encoding via
    #: ``ComponentIdentifier.to_dict()`` is the only sound round-trip path.
    GRAPH_ARTIFACT_OPAQUE: ClassVar[bool] = True

    def __init__(
        self,
        *,
        atomic_attack_name: str,
        atomic_attacks: Sequence[AtomicAttack],
        weak_categories_ref: Callable[[], set[str]],
        max_concurrency: int = 1,
    ) -> None:
        """
        Initialize the filtered deep-dive step.

        Args:
            atomic_attack_name (str): Step name for resume / diagnostics.
            atomic_attacks (Sequence[AtomicAttack]): The candidate atomics
                to run conditionally.
            weak_categories_ref (Callable[[], set[str]]): Callable that
                returns the live set of weak categories at dispatch time.
                Wrapping the lookup in a callable (rather than passing a
                set by reference) lets the sweep step build its weak-set
                fresh on each scenario attempt without the deep-dive step
                holding a stale reference.
            max_concurrency (int): Forwarded to each ``AtomicAttack.run_async``.

        Raises:
            ValueError: If ``atomic_attacks`` is empty.
        """
        if not atomic_attacks:
            raise ValueError("FilteredDeepDiveStep requires at least one atomic attack.")

        self.name = atomic_attack_name
        self.outputs = list(self._OUTPUTS)
        self._atomics = list(atomic_attacks)
        self._weak_categories_ref = weak_categories_ref
        self._max_concurrency = max_concurrency

        self.atomic_attack_name = atomic_attack_name
        self.display_group = atomic_attack_name
        # Aggregate seed groups across all wrapped atomics so the orchestrator
        # sees a comprehensive view even when individual atomics will be skipped.
        self.seed_groups = [g for atomic in self._atomics for g in atomic.seed_groups]
        seen_objectives: set[str] = set()
        self.objectives = []
        for atomic in self._atomics:
            for objective in atomic.objectives:
                if objective not in seen_objectives:
                    self.objectives.append(objective)
                    seen_objectives.add(objective)

    def filter_seed_groups_by_objectives(self, *, remaining_objectives: list[str]) -> None:
        """
        Forward objective filtering to each wrapped atomic attack.

        Args:
            remaining_objectives: Remaining objectives the orchestrator
                wants this step to cover.
        """
        for atomic in self._atomics:
            atomic.filter_seed_groups_by_objectives(remaining_objectives=remaining_objectives)
        self.seed_groups = [g for atomic in self._atomics for g in atomic.seed_groups]
        self.objectives = list(remaining_objectives)

    async def process_async(self) -> ScenarioStepResult:
        """
        Run each wrapped atomic conditionally on its category being flagged.

        Returns:
            ScenarioStepResult: Outcome is always ``"done"``. The aggregated
                ``attack_results`` contain results from every atomic that
                was actually dispatched. ``metadata['skipped_categories']``
                lists categories that were not in the weak set;
                ``metadata['dispatched_categories']`` lists the categories
                actually exercised.
        """
        weak = self._weak_categories_ref()
        attack_results: list[AttackResult] = []
        dispatched: list[str] = []
        skipped: list[str] = []

        for atomic in self._atomics:
            category = atomic.display_group or atomic.atomic_attack_name
            if category not in weak:
                skipped.append(category)
                continue
            executor_result = await atomic.run_async(
                max_concurrency=self._max_concurrency,
                return_partial_on_failure=True,
            )
            attack_results.extend(executor_result.completed_results)
            dispatched.append(category)

        return ScenarioStepResult(
            outcome="done",
            attack_results=attack_results,
            metadata={
                "dispatched_categories": dispatched,
                "skipped_categories": skipped,
            },
        )

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build the behavioral identity for the deep-dive step.

        Nests each wrapped atomic's identifier under ``children["deep_dive_attacks"]``
        (sorted by ``atomic_attack_name`` for deterministic hash stability).

        Returns:
            ComponentIdentifier: The frozen identity snapshot.
        """
        from pyrit.identifiers import ComponentIdentifier

        nested_ids = [atomic.get_identifier() for atomic in sorted(self._atomics, key=lambda a: a.atomic_attack_name)]
        return ComponentIdentifier.of(
            self,
            params={
                "atomic_attack_name": self.atomic_attack_name,
                "outputs": list(self.outputs),
            },
            children={"deep_dive_attacks": nested_ids},
        )


class BroadSweepThenDeepDive(Scenario):
    """
    Branching scenario: sweep with fast attacks, escalate only on weakness.

    Composes two phases into a non-linear ``StrategyGraph``:

    - **Sweep**: a ``CategoryAggregatingSweepStep`` runs one provided
      single-turn ``AtomicAttack`` across every seed group and reports
      which categories the model breached.
    - **Deep dive**: a ``FilteredDeepDiveStep`` runs the provided multi-turn
      ``AtomicAttack`` list — but only for categories the sweep flagged.

    The policy short-circuits to ``ALL_SAFE`` when the sweep finds no
    weaknesses, so the deep-dive step is never invoked. This is the Phase 7
    validation artifact that the branching graph abstraction actually
    expresses scenarios that today cannot be cleanly written with the
    flat-loop pattern.
    """

    VERSION: int = 1

    #: The branching graph in ``_build_execution_graph`` ignores the orchestrator-supplied
    #: ``steps`` list and dispatches a fixed sweep -> deep-dive policy, so an implicitly
    #: prepended baseline ``AtomicAttack`` would be persisted in ``_atomic_attacks`` but
    #: never executed by the graph. ``Forbidden`` makes the (correct) intent explicit and
    #: fails fast if a caller passes ``include_baseline=True``.
    BASELINE_ATTACK_POLICY: ClassVar[BaselineAttackPolicy] = BaselineAttackPolicy.Forbidden

    @apply_defaults
    def __init__(
        self,
        *,
        sweep_atomic_attack: AtomicAttack,
        deep_dive_atomic_attacks: Sequence[AtomicAttack],
        outcome_scorer: OutcomeScorer,
        weakness_label: str = "safety_violation",
        objective_scorer: Optional[TrueFalseScorer] = None,
        scenario_result_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the branching scenario.

        Args:
            sweep_atomic_attack (AtomicAttack): A single fast attack run
                across all seed groups during the sweep phase.
            deep_dive_atomic_attacks (Sequence[AtomicAttack]): The atomics
                to consider during the deep-dive phase. Each is gated by
                its ``display_group`` matching a category flagged by the
                sweep step.
            outcome_scorer (OutcomeScorer): Wraps the per-response classifier
                whose output labels each sweep result. Must declare
                ``weakness_label`` as one of its outputs.
            weakness_label (str): The label emitted by ``outcome_scorer``
                that signals a category breach. Defaults to
                ``"safety_violation"``.
            objective_scorer (TrueFalseScorer | None): Forwarded to the
                base ``Scenario``. Defaults to ``outcome_scorer.wrapped_scorer``
                cast to ``TrueFalseScorer`` so dataset config bootstrap
                stays consistent.
            scenario_result_id (str | None): Optional ID of an existing
                scenario result to resume.

        Raises:
            ValueError: If ``deep_dive_atomic_attacks`` is empty.
            ValueError: If ``weakness_label`` is not declared as one of
                ``outcome_scorer.outcomes``.
        """
        if not deep_dive_atomic_attacks:
            raise ValueError("BroadSweepThenDeepDive requires at least one deep_dive_atomic_attack.")

        # Fail fast: the inner ``CategoryAggregatingSweepStep`` performs the
        # same check, but only inside ``_build_execution_graph`` (called from
        # ``run_async``). A wizard-built scenario would otherwise serialize a
        # graph artifact and only error at first execution, defeating the
        # "validate at the surface that elicited the input" guarantee that
        # ``input_schema`` advertises for the ``weakness_label`` / ``outcome_scorer``
        # pair.
        if weakness_label not in outcome_scorer.outcomes:
            raise ValueError(
                f"weakness_label {weakness_label!r} is not declared as an outcome of the "
                f"supplied OutcomeScorer (declared: {outcome_scorer.outcomes!r})."
            )

        self._sweep_atomic = sweep_atomic_attack
        self._deep_dive_atomics: list[AtomicAttack] = list(deep_dive_atomic_attacks)
        self._outcome_scorer = outcome_scorer
        self._weakness_label = weakness_label

        # Shared mutable handle the sweep step updates and the deep-dive step
        # reads. Reset on each ``run_async`` via ``_build_execution_graph``.
        self._weak_categories: set[str] = set()

        effective_objective_scorer = (
            objective_scorer if objective_scorer is not None else cast("TrueFalseScorer", outcome_scorer.wrapped_scorer)
        )

        super().__init__(
            version=self.VERSION,
            objective_scorer=effective_objective_scorer,
            strategy_class=self.get_strategy_class(),
            scenario_result_id=scenario_result_id,
        )

    @classmethod
    def input_schema(cls) -> list[RoleDescriptor]:
        """
        Declare the rich-object and scalar inputs the wizard / artifact must capture.

        The three opaque roles are pre-built ``Identifiable`` instances that the
        CLI wizard cannot elicit directly — programmatic callers must supply them
        and CLI flows must round-trip them through a saved graph artifact (see
        :class:`pyrit.scenario.core.graph_artifact.GraphArtifact`). The single
        scalar role (``weakness_label``) is freely elicitable.

        Returns:
            list[RoleDescriptor]: Three OPAQUE roles plus one SCALAR.
        """
        return [
            RoleDescriptor(
                name="sweep_atomic_attack",
                description=(
                    "Pre-built single-turn AtomicAttack run across all seed groups during the sweep phase. "
                    "Must already be wired to a target and scorer."
                ),
                tag=RoleTag.OPAQUE,
                required=True,
            ),
            RoleDescriptor(
                name="deep_dive_atomic_attacks",
                description=(
                    "Pre-built sequence of multi-turn AtomicAttacks considered during the deep-dive phase. "
                    "Each is gated by its ``display_group`` matching a category flagged by the sweep."
                ),
                tag=RoleTag.OPAQUE,
                required=True,
            ),
            RoleDescriptor(
                name="outcome_scorer",
                description=(
                    "Pre-built OutcomeScorer whose per-response label set must include ``weakness_label``. "
                    "Drives the sweep's category-weakness classification."
                ),
                tag=RoleTag.OPAQUE,
                required=True,
            ),
            RoleDescriptor(
                name="weakness_label",
                description=(
                    "OutcomeScorer label that signals a category breach and triggers escalation to deep dive."
                ),
                tag=RoleTag.SCALAR,
                param_type=str,
                default="safety_violation",
                required=False,
            ),
        ]

    @classmethod
    def get_strategy_class(cls) -> type[ScenarioStrategy]:
        """Return the (single-member) strategy enum class."""
        return BroadSweepThenDeepDiveStrategy

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        """Return the only strategy member."""
        return BroadSweepThenDeepDiveStrategy.DEFAULT

    @classmethod
    def default_dataset_config(cls) -> DatasetConfiguration:
        """
        Return an empty dataset configuration.

        The atomics are supplied directly via the constructor, so the
        base scenario's auto-build from registered datasets is unused.

        Returns:
            DatasetConfiguration: An empty configuration; this scenario
                does not consume registered datasets.
        """
        return DatasetConfiguration()

    async def _get_atomic_attacks_async(self) -> list[AtomicAttack]:
        """
        Return the atomics in canonical phase order: sweep first, deep dives after.

        The orchestrator uses this list as the resume-by-name keyspace. We
        keep the order stable so resume can rehydrate deterministically.

        Returns:
            list[AtomicAttack]: ``[sweep_atomic, *deep_dive_atomics]``.
        """
        return [self._sweep_atomic, *self._deep_dive_atomics]

    def _get_attack_technique_factories(self) -> dict[str, AttackTechniqueFactory]:
        """
        Return an empty factory map: this scenario does not use the technique registry.

        The sweep and deep-dive atomics are supplied directly via the constructor,
        so factory lookup never happens during execution. Overriding the base
        method (which lazily populates the global ``AttackTechniqueRegistry``
        singleton via ``register_scenario_techniques``) keeps introspection
        (e.g. :func:`pyrit.scenario.core.waterfall.policy_to_spec`) side-effect-free
        and makes the policy-parameterized intent explicit, mirroring the
        ``BASELINE_ATTACK_POLICY = Forbidden`` declaration above.

        Returns:
            dict[str, AttackTechniqueFactory]: Always empty.
        """
        return {}

    def _build_execution_graph(  # ty: ignore[invalid-method-override]
        self,
        *,
        steps: Optional[Sequence[ScenarioStep]] = None,
    ) -> StrategyGraph[ScenarioStep, SweepThenDeepDiveState]:
        """
        Build the branching graph that drives the sweep → deep-dive flow.

        Ignores the orchestrator-supplied ``steps`` argument: this scenario
        owns the partitioning of ``self._atomic_attacks`` into the sweep
        and deep-dive phases and cannot be driven by a flat step list. If
        the resume filter removes the sweep atomic (because it already
        completed), the sweep step's wrapped attack will return its
        previously-completed results from memory; the policy still runs
        the classification pass to populate ``self._weak_categories``.

        Note:
            This override widens ``StateT`` from ``int`` (the base-class
            default for the linear policy) to a per-scenario enum. The
            base method is an extension point, but the static return type
            cannot be expressed without making ``Scenario`` generic over
            ``StateT``. Long-term fix tracked as a Phase 9 cleanup; the
            runtime contract (``StrategyGraph[ScenarioStep, Any]``) is
            preserved. See plan.md for context.

        Args:
            steps: Ignored. Present only to honor the base-class signature.

        Returns:
            StrategyGraph[ScenarioStep, SweepThenDeepDiveState]: The branching
                graph instance.
        """
        # Reset shared mutable state for this attempt.
        self._weak_categories = set()

        sweep_step = CategoryAggregatingSweepStep(
            atomic_attack_name=self._sweep_atomic.atomic_attack_name,
            atomic_attack=self._sweep_atomic,
            outcome_scorer=self._outcome_scorer,
            weakness_label=self._weakness_label,
            max_concurrency=self._max_concurrency,
        )
        deep_dive_step = FilteredDeepDiveStep(
            atomic_attack_name=f"{self._sweep_atomic.atomic_attack_name}_deep_dive",
            atomic_attacks=self._deep_dive_atomics,
            weak_categories_ref=lambda: self._weak_categories,
            max_concurrency=self._max_concurrency,
        )

        policy = self._build_branching_policy(sweep_step=sweep_step, deep_dive_step=deep_dive_step)
        return StrategyGraph(policy=policy)

    def _build_branching_policy(
        self,
        *,
        sweep_step: CategoryAggregatingSweepStep,
        deep_dive_step: FilteredDeepDiveStep,
    ) -> StrategyPolicy[ScenarioStep, SweepThenDeepDiveState]:
        """
        Compose the two-action policy for the branching graph.

        States:
          - ``SWEEPING`` → ``_sweep_action`` runs the sweep step. Emits
            ``DEEP_DIVING`` if ``"found_weaknesses"`` was the sweep
            outcome, else ``ALL_SAFE`` (terminal).
          - ``DEEP_DIVING`` → ``_deep_dive_action`` runs the deep-dive
            step. Always emits ``COMPLETE`` (terminal).
          - ``COMPLETE`` and ``ALL_SAFE`` are terminal states.

        Args:
            sweep_step: The sweep step to dispatch from ``SWEEPING``.
            deep_dive_step: The deep-dive step to dispatch from
                ``DEEP_DIVING``.

        Returns:
            StrategyPolicy: A frozen policy with the two-state branching
                action map.
        """

        async def _sweep_action(
            graph: StrategyGraph[ScenarioStep, SweepThenDeepDiveState],
        ) -> tuple[SweepThenDeepDiveState, ScenarioStepResult | None]:
            graph.bind_current_step(step=sweep_step)
            try:
                base_result = await sweep_step.process_async()
                # Update the closure-shared weak-categories set so the
                # deep-dive step sees the fresh classification.
                self._weak_categories = set(base_result.metadata.get("weak_categories", set()))
                merged_metadata = {"step_name": sweep_step.name, **base_result.metadata}
                result = ScenarioStepResult(
                    outcome=base_result.outcome,
                    attack_results=list(base_result.attack_results),
                    step_identifier=base_result.step_identifier,
                    metadata=merged_metadata,
                )
            finally:
                graph.bind_current_step(step=None)

            next_state = (
                SweepThenDeepDiveState.DEEP_DIVING
                if base_result.outcome == "found_weaknesses"
                else SweepThenDeepDiveState.ALL_SAFE
            )
            return next_state, result

        async def _deep_dive_action(
            graph: StrategyGraph[ScenarioStep, SweepThenDeepDiveState],
        ) -> tuple[SweepThenDeepDiveState, ScenarioStepResult | None]:
            graph.bind_current_step(step=deep_dive_step)
            try:
                base_result = await deep_dive_step.process_async()
                merged_metadata = {"step_name": deep_dive_step.name, **base_result.metadata}
                result = ScenarioStepResult(
                    outcome=base_result.outcome,
                    attack_results=list(base_result.attack_results),
                    step_identifier=base_result.step_identifier,
                    metadata=merged_metadata,
                )
            finally:
                graph.bind_current_step(step=None)
            return SweepThenDeepDiveState.COMPLETE, result

        actions: dict[SweepThenDeepDiveState, PolicyAction[ScenarioStep, SweepThenDeepDiveState]] = {
            SweepThenDeepDiveState.SWEEPING: _sweep_action,
            SweepThenDeepDiveState.DEEP_DIVING: _deep_dive_action,
        }

        return StrategyPolicy(
            actions=actions,
            initial_state=SweepThenDeepDiveState.SWEEPING,
            terminal_states=frozenset({SweepThenDeepDiveState.COMPLETE, SweepThenDeepDiveState.ALL_SAFE}),
        )
