# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``AdaptiveDispatchAttack`` — picks an inner technique per attempt via an
``AdaptiveTechniqueSelector``, runs it, records the outcome, and loops up to
``max_attempts_per_objective`` times. Reads the per-objective context key from
``context.memory_labels[ADAPTIVE_CONTEXT_LABEL]`` (falls back to the global context).

The dispatcher is bound to a single ``SeedAttackGroup`` at construction time so
it can merge each chosen technique's ``seed_technique`` (when present) into the
seed group before delegating execution to ``AttackExecutor``.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pyrit.executor.attack.core.attack_executor import AttackExecutor
from pyrit.executor.attack.core.attack_parameters import AttackParameters
from pyrit.executor.attack.core.attack_strategy import AttackContext, AttackStrategy
from pyrit.models import AttackOutcome, AttackResult
from pyrit.scenario.scenarios.adaptive.selector import (
    GLOBAL_CONTEXT,
    AdaptiveTechniqueSelector,
)

if TYPE_CHECKING:
    from pyrit.models import SeedAttackGroup, SeedAttackTechniqueGroup
    from pyrit.prompt_target import PromptTarget
    from pyrit.score import TrueFalseScorer

logger = logging.getLogger(__name__)


# Memory-label keys stamped onto persisted prompt rows so adaptive attempts
# can be filtered/grouped after a run. The scenario stamps the context once
# per objective; the dispatcher stamps technique + attempt index on each try.
ADAPTIVE_CONTEXT_LABEL: str = "_adaptive_context"
"""Per-objective context key (e.g. ``"_global"`` or a harm category)."""
ADAPTIVE_TECHNIQUE_LABEL: str = "_adaptive_technique"
"""Technique chosen by the dispatcher for a given attempt."""
ADAPTIVE_ATTEMPT_LABEL: str = "_adaptive_attempt"
"""1-based attempt index within the per-objective loop."""


@dataclass(frozen=True)
class TechniqueBundle:
    """
    Per-technique bundle consumed by the dispatcher.

    Carries the inner attack strategy alongside the factory-supplied
    ``seed_technique`` (if any) and ``adversarial_chat`` (required when the
    seed_technique contains a simulated-conversation config).
    """

    attack: AttackStrategy[Any, AttackResult]
    seed_technique: SeedAttackTechniqueGroup | None = None
    adversarial_chat: PromptTarget | None = None


@dataclass
class AdaptiveDispatchContext(AttackContext[AttackParameters]):
    """Execution context for ``AdaptiveDispatchAttack`` (no extra state)."""


class AdaptiveDispatchAttack(AttackStrategy[AdaptiveDispatchContext, AttackResult]):
    """
    Attack that delegates each attempt to one of several inner techniques,
    choosing per attempt via an ``AdaptiveTechniqueSelector``.

    For each objective, loops up to ``max_attempts_per_objective`` times:
    ask the selector, execute the chosen technique, record the outcome, and
    stop early on success. The selector is shared by reference with the
    scenario so learning accumulates across objectives.

    The dispatcher is bound to a single ``SeedAttackGroup`` at construction
    time. When a chosen technique declares a ``seed_technique``, that group
    is merged into the seed group before execution (mirroring the static
    ``AtomicAttack`` path).

    On success, the dispatcher returns a fresh ``AttackResult`` copy of the
    winning inner result (new ``attack_result_id`` and ``timestamp``) with
    the dispatch trail stamped onto ``metadata``. The inner result has
    already been persisted by its own post-execute hook, so two rows are
    written per successful objective sharing the same ``conversation_id``:
    the inner row carries the raw outcome, the outer row carries the
    adaptive trail.
    """

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        techniques: dict[str, TechniqueBundle],
        selector: AdaptiveTechniqueSelector,
        seed_group: SeedAttackGroup,
        objective_scorer: TrueFalseScorer | None = None,
        max_attempts_per_objective: int = 3,
    ) -> None:
        """
        Args:
            objective_target (PromptTarget): The target inner attacks run against.
                Stored for identifier/logging parity; not called directly.
            techniques (dict[str, TechniqueBundle]): Mapping from technique name to
                its bundle (attack, seed_technique, adversarial_chat). Must be non-empty.
            selector (AdaptiveTechniqueSelector): Shared selector state.
            seed_group (SeedAttackGroup): The seed group bound to this dispatcher.
                Each attempt's chosen technique is applied against this group
                (merging the technique's ``seed_technique`` when present).
            objective_scorer (TrueFalseScorer | None): Scorer passed through to
                techniques that generate simulated conversations.
            max_attempts_per_objective (int): Max attempts per objective; >= 1.
                Defaults to 3.

        Raises:
            ValueError: If ``techniques`` is empty or ``max_attempts_per_objective`` < 1.
        """
        if not techniques:
            raise ValueError("techniques must contain at least one attack technique")
        if max_attempts_per_objective < 1:
            raise ValueError(f"max_attempts_per_objective must be >= 1, got {max_attempts_per_objective}")

        super().__init__(
            objective_target=objective_target,
            context_type=AdaptiveDispatchContext,
            params_type=AttackParameters,
            logger=logger,
        )
        self._techniques = techniques
        self._selector = selector
        self._seed_group = seed_group
        self._objective_scorer = objective_scorer
        self._max_attempts = max_attempts_per_objective
        # Attempts are inherently sequential (each one reads the selector
        # state updated by the previous), so a single shared executor with
        # ``max_concurrency=1`` is reused across attempts.
        self._executor = AttackExecutor(max_concurrency=1)

    def _validate_context(self, *, context: AdaptiveDispatchContext) -> None:
        """
        Ensure the context carries a non-empty objective string.

        Raises:
            ValueError: If ``context.objective`` is empty or whitespace-only.
        """
        if not context.objective or context.objective.isspace():
            raise ValueError("Attack objective must be provided and non-empty")

    async def _setup_async(self, *, context: AdaptiveDispatchContext) -> None:
        """No-op: per-attempt setup is owned by the inner technique's executor."""

    async def _teardown_async(self, *, context: AdaptiveDispatchContext) -> None:
        """No-op: per-attempt teardown is owned by the inner technique's executor."""

    async def _run_inner_attack_async(
        self,
        *,
        bundle: TechniqueBundle,
        attempt_labels: dict[str, str],
    ) -> AttackResult:
        """
        Execute the chosen technique against this dispatcher's seed group.

        Merges ``bundle.seed_technique`` into the bound ``seed_group`` (when
        present) and delegates execution to ``AttackExecutor``. Isolated as a
        method so tests can patch the inner-attack call surface.

        Args:
            bundle (TechniqueBundle): The chosen technique's attack + seeds + chat.
            attempt_labels (dict[str, str]): Memory labels stamped onto this attempt.

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

    async def _perform_async(self, *, context: AdaptiveDispatchContext) -> AttackResult:
        """
        Run the per-objective adaptive loop.

        Resolves the per-objective context key from ``context.memory_labels``
        (falling back to :data:`GLOBAL_CONTEXT`), then loops up to
        ``max_attempts_per_objective`` times: select a technique, execute it,
        record the outcome, and stop early on success.

        Args:
            context (AdaptiveDispatchContext): Execution context. ``memory_labels``
                may carry :data:`ADAPTIVE_CONTEXT_LABEL` to scope the selector.

        Returns:
            AttackResult: A fresh dispatcher-owned copy of the final inner
                result with the dispatch trail stamped onto ``metadata``
                (see class docstring for the two-row persistence note).

        Raises:
            RuntimeError: If the loop somehow ran zero attempts (unreachable
                because ``max_attempts_per_objective`` is validated >= 1).
        """
        adaptive_context = context.memory_labels.get(ADAPTIVE_CONTEXT_LABEL, GLOBAL_CONTEXT)
        technique_names = list(self._techniques.keys())

        last_result: AttackResult | None = None
        trail: list[dict[str, str]] = []

        for attempt_idx in range(self._max_attempts):
            chosen = self._selector.select(context=adaptive_context, techniques=technique_names)
            bundle = self._techniques[chosen]
            attempt_labels = {
                **context.memory_labels,
                ADAPTIVE_TECHNIQUE_LABEL: chosen,
                ADAPTIVE_ATTEMPT_LABEL: str(attempt_idx + 1),
            }

            logger.debug(
                "AdaptiveDispatchAttack: attempt %d/%d context=%r technique=%r",
                attempt_idx + 1,
                self._max_attempts,
                adaptive_context,
                chosen,
            )

            result = await self._run_inner_attack_async(bundle=bundle, attempt_labels=attempt_labels)
            success = result.outcome == AttackOutcome.SUCCESS
            self._selector.record_outcome(context=adaptive_context, technique=chosen, success=success)

            trail.append({"technique": chosen, "outcome": result.outcome.value})
            last_result = result

            if success:
                break

        # ``max_attempts`` is validated >= 1, so the loop always runs at least
        # once. Guard explicitly rather than with ``assert`` (stripped under -O).
        if last_result is None:  # pragma: no cover - defensive
            raise RuntimeError("AdaptiveDispatchAttack ran zero attempts; this should be unreachable.")
        # Return a fresh dispatcher-owned ``AttackResult``: the inner attack
        # already persisted ``last_result`` via its own post-execute hook, so
        # returning it directly would cause a PK conflict on the outer hook.
        # ``dataclasses.replace`` copies every field; we override identity
        # fields and stamp the trail onto metadata.
        return replace(
            last_result,
            attack_result_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={
                **last_result.metadata,
                "adaptive_attempts": trail,
                "adaptive_context": adaptive_context,
            },
        )
