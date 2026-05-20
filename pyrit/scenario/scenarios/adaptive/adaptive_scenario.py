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

import logging
import random
import uuid
from typing import TYPE_CHECKING, ClassVar

from pyrit.executor.attack import AttackScoringConfig
from pyrit.scenario.core.atomic_attack import AtomicAttack
from pyrit.scenario.core.attack_technique import AttackTechnique
from pyrit.scenario.core.scenario import BaselinePolicy, Scenario
from pyrit.scenario.scenarios.adaptive.dispatcher import (
    ADAPTIVE_CONTEXT_LABEL,
    AdaptiveDispatchAttack,
    TechniqueBundle,
)
from pyrit.scenario.scenarios.adaptive.selector import (
    AdaptiveTechniqueSelector,
    ContextExtractor,
    global_context,
)

if TYPE_CHECKING:
    from pyrit.models import SeedAttackGroup
    from pyrit.prompt_target import PromptTarget
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

    BASELINE_POLICY: ClassVar[BaselinePolicy] = BaselinePolicy.Forbidden

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

        super().__init__(
            version=self.VERSION,
            strategy_class=self.get_strategy_class(),
            objective_scorer=objective_scorer,
            scenario_result_id=scenario_result_id,
        )

    async def _get_atomic_attacks_async(self) -> list[AtomicAttack]:
        """
        Build one ``AtomicAttack`` per objective.

        Each objective gets a freshly constructed ``AdaptiveDispatchAttack``
        bound to its seed group, but all dispatchers share the same selector
        so learning accumulates across objectives. Per-objective, techniques
        whose ``seed_technique`` is incompatible with the seed group are
        filtered out; objectives left with no compatible techniques are skipped.

        Returns:
            list[AtomicAttack]: One ``AtomicAttack`` per objective with at
                least one compatible technique. Empty if every seed group
                is incompatible with every selected technique.

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

        seed_groups_by_dataset = self._dataset_config.get_seed_attack_groups()
        atomic_attacks: list[AtomicAttack] = []
        for dataset_name, seed_groups in seed_groups_by_dataset.items():
            for seed_group in seed_groups:
                atomic = self._build_atomic_for_seed_group(
                    dataset_name=dataset_name,
                    seed_group=seed_group,
                    techniques=techniques,
                    selector=selector,
                )
                if atomic is not None:
                    atomic_attacks.append(atomic)

        return atomic_attacks

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

    def _build_atomic_for_seed_group(
        self,
        *,
        dataset_name: str,
        seed_group: SeedAttackGroup,
        techniques: dict[str, TechniqueBundle],
        selector: AdaptiveTechniqueSelector,
    ) -> AtomicAttack | None:
        """
        Build a single ``AtomicAttack`` for one ``SeedAttackGroup``.

        Filters the technique pool down to those whose ``seed_technique`` (if
        any) is compatible with this seed group, then constructs a dedicated
        ``AdaptiveDispatchAttack`` bound to this seed group.

        Returns:
            AtomicAttack | None: The constructed atomic attack, or ``None`` when
                no techniques are compatible (caller skips the objective).

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
        # Prefer the objective's id when available so resume keys stay stable
        # across re-fetches of the same seed groups.
        objective_id = seed_group.objective.id if seed_group.objective.id else uuid.uuid4()
        atomic_attack_name = f"{self._atomic_attack_prefix}_{dataset_name}_{objective_id}"

        dispatcher = AdaptiveDispatchAttack(
            objective_target=self._objective_target,
            techniques=compatible,
            selector=selector,
            seed_group=seed_group,
            objective_scorer=self._objective_scorer,
            max_attempts_per_objective=self._max_attempts_per_objective,
        )

        memory_labels = {
            **self._memory_labels,
            ADAPTIVE_CONTEXT_LABEL: adaptive_context,
        }
        return AtomicAttack(
            atomic_attack_name=atomic_attack_name,
            attack_technique=AttackTechnique(attack=dispatcher),
            seed_groups=[seed_group],
            objective_scorer=self._objective_scorer,
            memory_labels=memory_labels,
            display_group=dataset_name,
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
