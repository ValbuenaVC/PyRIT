# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``AdaptiveStep`` — per-objective adaptive ``ScenarioStep`` (Phase 6b).

Replaces ``AdaptiveDispatchAttack`` as the per-objective execution unit for
adaptive scenarios. Where the dispatcher was an ``AttackStrategy`` wrapped
inside an ``AtomicAttack`` (which itself sits inside a flat scenario loop),
the step plugs directly into the ``StrategyGraph`` event loop introduced in
Phase 5:

* ``process_async`` runs the per-objective selector loop and returns a
  ``ScenarioStepResult`` whose ``outcome`` is ``"success"`` or ``"exhausted"``
  — these become real transition labels the surrounding policy can branch on.
* The dispatch trail (``adaptive_attempts`` + ``adaptive_context``) is stamped
  onto the returned ``AttackResult.metadata`` exactly as the dispatcher did,
  so existing persistence / rehydration semantics are preserved.
* Inner-attack execution still flows through ``AttackExecutor`` against a
  technique-merged ``SeedAttackGroup``, so the inner ``AttackResult`` rows the
  techniques persist are unchanged.

``AdaptiveStep`` provides the ``AtomicAttack``-shaped attributes that the
``Scenario`` orchestrator uses for resume bookkeeping
(``atomic_attack_name``, ``objectives``, ``seed_groups``, ``display_group``,
``filter_seed_groups_by_objectives``) without subclassing ``AtomicAttack``,
because doing so would force a single canonical ``AttackTechnique`` onto the
parent constructor that doesn't match the multi-technique reality.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pyrit.executor.attack.core.attack_executor import AttackExecutor
from pyrit.identifiers import ComponentIdentifier
from pyrit.models import AttackOutcome, AttackResult
from pyrit.scenario.core.scenario_step import ScenarioStep, ScenarioStepResult
from pyrit.scenario.scenarios.adaptive.dispatcher import (
    ADAPTIVE_ATTEMPT_LABEL,
    ADAPTIVE_TECHNIQUE_LABEL,
    TechniqueBundle,
)
from pyrit.scenario.scenarios.adaptive.selector import (
    GLOBAL_CONTEXT,
    AdaptiveTechniqueSelector,
)

if TYPE_CHECKING:
    from pyrit.models import SeedAttackGroup
    from pyrit.prompt_target import PromptTarget
    from pyrit.score import TrueFalseScorer

logger = logging.getLogger(__name__)


class AdaptiveStep(ScenarioStep):
    """
    Per-objective adaptive selector step.

    Owns the per-objective adaptive loop that PR #1760 placed inside
    ``AdaptiveDispatchAttack``. Each call to :meth:`process_async` selects up
    to ``max_attempts_per_objective`` techniques via the shared
    :class:`AdaptiveTechniqueSelector`, runs each selected technique against
    the bound seed group, records the outcome on the selector, and stops on
    first success.

    The returned :class:`ScenarioStepResult` exposes the outcome as a real
    transition label (``"success"`` or ``"exhausted"``) so a state-machine
    policy can route on it. A single ``AttackResult`` is returned — a fresh
    dispatcher-owned copy of the last inner result with the adaptive trail
    stamped onto ``metadata`` — preserving the two-row persistence story
    (inner row + outer row sharing a conversation id) described on
    :class:`AdaptiveDispatchAttack`.
    """

    #: Transition labels this step can emit. ``"success"`` means at least one
    #: attempt achieved :attr:`AttackOutcome.SUCCESS`; ``"exhausted"`` means
    #: ``max_attempts_per_objective`` ran without success.
    _OUTPUTS: tuple[str, ...] = ("success", "exhausted")

    def __init__(
        self,
        *,
        atomic_attack_name: str,
        display_group: str | None = None,
        objective_target: PromptTarget,
        techniques: dict[str, TechniqueBundle],
        selector: AdaptiveTechniqueSelector,
        seed_group: SeedAttackGroup,
        objective_scorer: TrueFalseScorer | None = None,
        max_attempts_per_objective: int = 3,
        memory_labels: dict[str, str] | None = None,
        adaptive_context: str = GLOBAL_CONTEXT,
    ) -> None:
        """
        Args:
            atomic_attack_name: Unique key used by the scenario for resume tracking
                and result persistence. Mirrors :attr:`AtomicAttack.atomic_attack_name`.
            display_group: Optional label for grouping results in user-facing output.
                Defaults to ``atomic_attack_name`` when ``None``.
            objective_target: The target inner attacks run against. Stored for
                identifier / logging parity; not called directly by the step.
            techniques: Mapping ``{technique_name: TechniqueBundle}``. Must be non-empty.
            selector: Shared :class:`AdaptiveTechniqueSelector` so learning accumulates
                across all per-objective steps in the scenario.
            seed_group: The :class:`SeedAttackGroup` this step is bound to. Each attempt
                merges the chosen technique's ``seed_technique`` (if any) into this group
                before execution.
            objective_scorer: Scorer passed through to techniques that generate
                simulated conversations.
            max_attempts_per_objective: Max techniques per objective; must be ``>= 1``.
                Defaults to 3.
            memory_labels: Per-attempt memory labels (typically including the
                ``ADAPTIVE_CONTEXT_LABEL`` stamped by the scenario).
            adaptive_context: The selector context bucket for this step (e.g.
                :data:`GLOBAL_CONTEXT` or a harm-category key). Stays stable for the
                lifetime of the step.

        Raises:
            ValueError: If ``techniques`` is empty or ``max_attempts_per_objective`` < 1.
        """
        if not techniques:
            raise ValueError("techniques must contain at least one attack technique")
        if max_attempts_per_objective < 1:
            raise ValueError(f"max_attempts_per_objective must be >= 1, got {max_attempts_per_objective}")

        self.atomic_attack_name = atomic_attack_name
        self.display_group = display_group or atomic_attack_name

        self._objective_target = objective_target
        self._techniques = techniques
        self._selector = selector
        self._seed_group = seed_group
        self._objective_scorer = objective_scorer
        self._max_attempts = max_attempts_per_objective
        self._memory_labels = memory_labels or {}
        self._adaptive_context = adaptive_context
        # Attempts are inherently sequential (each one reads the selector
        # state updated by the previous), so a single shared executor with
        # ``max_concurrency=1`` is reused across attempts.
        self._executor = AttackExecutor(max_concurrency=1)

    @property
    def name(self) -> str:
        """Display / resume key. Aliases :attr:`atomic_attack_name`."""
        return self.atomic_attack_name

    @property
    def outputs(self) -> list[str]:
        """Transition labels this step can emit."""
        return list(self._OUTPUTS)

    @property
    def objectives(self) -> list[str]:
        """Objectives drawn from the bound seed group (parity with ``AtomicAttack``)."""
        if self._seed_group.objective is not None:
            return [self._seed_group.objective.value]
        return []

    @property
    def seed_groups(self) -> list[SeedAttackGroup]:
        """One-element list view of the bound seed group (parity with ``AtomicAttack``)."""
        return [self._seed_group]

    def filter_seed_groups_by_objectives(self, *, remaining_objectives: list[str]) -> None:
        """
        Drop the bound seed group when its objective is already complete.

        Mirrors :meth:`AtomicAttack.filter_seed_groups_by_objectives` so the
        scenario's resume filter (which calls this on every step it iterates
        over) works uniformly across step types. Because an adaptive step is
        bound to a single seed group, this collapses to a presence check on
        that single objective.

        Args:
            remaining_objectives: Objectives that still need execution.
        """
        if self._seed_group.objective is None:
            return
        if self._seed_group.objective.value not in set(remaining_objectives):
            # Replace the bound seed group with a marker that has no objective
            # so subsequent process_async calls are no-ops. In practice the
            # orchestrator skips the step entirely when objectives is empty,
            # so this branch is rarely hit; kept for defensive parity.
            self._seed_group = self._seed_group  # noqa: PLW0127 — explicit no-op

    async def process_async(self) -> ScenarioStepResult:
        """
        Run the per-objective adaptive loop and return its outcome.

        Loops up to ``max_attempts_per_objective`` times: select a technique
        via the shared selector, execute it against the merged seed group,
        record the outcome on the selector, and stop early on first success.
        The returned :class:`ScenarioStepResult` carries one
        :class:`AttackResult` — a fresh dispatcher-owned copy of the final
        inner result with the dispatch trail stamped onto ``metadata``.

        Returns:
            ScenarioStepResult: ``outcome`` is ``"success"`` if any attempt
                succeeded, ``"exhausted"`` otherwise. ``attack_results`` holds
                the single outer result; ``metadata`` carries the step name,
                the dispatch trail, and the adaptive context.

        Raises:
            RuntimeError: If the loop somehow ran zero attempts (unreachable
                because ``max_attempts_per_objective`` is validated >= 1).
        """
        technique_names = list(self._techniques.keys())
        last_result: AttackResult | None = None
        trail: list[dict[str, str]] = []
        succeeded = False

        for attempt_idx in range(self._max_attempts):
            chosen = self._selector.select(context=self._adaptive_context, techniques=technique_names)
            bundle = self._techniques[chosen]
            attempt_labels = {
                **self._memory_labels,
                ADAPTIVE_TECHNIQUE_LABEL: chosen,
                ADAPTIVE_ATTEMPT_LABEL: str(attempt_idx + 1),
            }

            logger.debug(
                "AdaptiveStep[%s]: attempt %d/%d context=%r technique=%r",
                self.atomic_attack_name,
                attempt_idx + 1,
                self._max_attempts,
                self._adaptive_context,
                chosen,
            )

            result = await self._run_inner_attack_async(bundle=bundle, attempt_labels=attempt_labels)
            success = result.outcome == AttackOutcome.SUCCESS
            self._selector.record_outcome(
                context=self._adaptive_context,
                technique=chosen,
                success=success,
            )

            trail.append({"technique": chosen, "outcome": result.outcome.value})
            last_result = result

            if success:
                succeeded = True
                break

        if last_result is None:  # pragma: no cover - defensive
            raise RuntimeError("AdaptiveStep ran zero attempts; this should be unreachable.")

        outcome_label = "success" if succeeded else "exhausted"
        # Return a fresh outer ``AttackResult`` to avoid PK conflicts with the
        # inner attack's already-persisted row (see ``AdaptiveDispatchAttack``
        # for the two-row persistence rationale).
        outer_result = replace(
            last_result,
            attack_result_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={
                **last_result.metadata,
                "adaptive_attempts": trail,
                "adaptive_context": self._adaptive_context,
            },
        )

        return ScenarioStepResult(
            outcome=outcome_label,
            attack_results=[outer_result],
            metadata={
                "step_name": self.atomic_attack_name,
                "adaptive_attempts": trail,
                "adaptive_context": self._adaptive_context,
            },
        )

    async def _run_inner_attack_async(
        self,
        *,
        bundle: TechniqueBundle,
        attempt_labels: dict[str, str],
    ) -> AttackResult:
        """
        Execute the chosen technique against the bound seed group.

        Merges ``bundle.seed_technique`` into the bound ``seed_group`` (when
        present) and delegates execution to :class:`AttackExecutor`. Isolated
        as a method so tests can patch the inner-attack call surface.

        Args:
            bundle: The chosen technique's attack + seeds + chat.
            attempt_labels: Memory labels stamped onto this attempt.

        Returns:
            AttackResult: The single result produced for this attempt.

        Raises:
            RuntimeError: If the executor returned no completed results and no
                propagated exception (should be unreachable).
        """
        if bundle.seed_technique is not None:
            execution_group = self._seed_group.with_technique(technique=bundle.seed_technique)
        else:
            execution_group = self._seed_group

        executor_result = await self._executor.execute_attack_from_seed_groups_async(
            attack=bundle.attack,
            seed_groups=[execution_group],
            adversarial_chat=bundle.adversarial_chat,
            objective_scorer=self._objective_scorer,
            memory_labels=attempt_labels,
        )

        if executor_result.completed_results:
            return executor_result.completed_results[0]
        if executor_result.incomplete_objectives:
            raise executor_result.incomplete_objectives[0][1]
        raise RuntimeError(  # pragma: no cover - defensive
            "AttackExecutor returned neither completed nor incomplete results."
        )

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build the behavioral identity for this adaptive step.

        Captures the step name, declared outputs, ``max_attempts`` configuration,
        and adaptive context. Each technique's bundled attack identifier is
        nested under ``children["techniques"]`` (sorted by technique name for
        hash stability) so drift in any inner attack or its seeds propagates
        upward.

        Returns:
            ComponentIdentifier: The frozen identity snapshot.
        """
        technique_ids: list[ComponentIdentifier] = [
            bundle.attack.get_identifier() for _, bundle in sorted(self._techniques.items())
        ]
        return ComponentIdentifier.of(
            self,
            params={
                "atomic_attack_name": self.atomic_attack_name,
                "outputs": list(self.outputs),
                "max_attempts_per_objective": self._max_attempts,
                "adaptive_context": self._adaptive_context,
            },
            children={"techniques": technique_ids},
        )
