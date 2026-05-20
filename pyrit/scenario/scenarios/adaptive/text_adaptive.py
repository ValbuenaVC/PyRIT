# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``TextAdaptive`` — text adaptive scenario.

Picks attack techniques per-objective using an epsilon-greedy selector
informed by observed success rates. Runs up to ``max_attempts_per_objective``
techniques per objective and stops early on success. The available techniques
come from the selected scenario strategies (``--strategies single_turn``
restricts to single-turn techniques, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from pyrit.common import apply_defaults
from pyrit.registry.tag_query import TagQuery
from pyrit.scenario.core.dataset_configuration import DatasetConfiguration
from pyrit.scenario.scenarios.adaptive.adaptive_scenario import AdaptiveScenario
from pyrit.scenario.scenarios.adaptive.selector import (
    ContextExtractor,
    global_context,
)

if TYPE_CHECKING:
    from pyrit.scenario.core.scenario_strategy import ScenarioStrategy
    from pyrit.score import TrueFalseScorer

logger = logging.getLogger(__name__)


def _build_text_adaptive_strategy() -> type[ScenarioStrategy]:
    """
    Build the strategy enum from the core scenario-techniques catalog.

    Returns:
        type[ScenarioStrategy]: The dynamically-built strategy enum class.
    """
    from pyrit.registry.object_registries.attack_technique_registry import (
        AttackTechniqueRegistry,
    )
    from pyrit.scenario.core.scenario_techniques import SCENARIO_TECHNIQUES

    return AttackTechniqueRegistry.build_strategy_class_from_specs(  # type: ignore[return-value, ty:invalid-return-type]
        class_name="TextAdaptiveStrategy",
        specs=SCENARIO_TECHNIQUES,
        aggregate_tags={
            "default": TagQuery.any_of("default"),
            "single_turn": TagQuery.any_of("single_turn"),
            "multi_turn": TagQuery.any_of("multi_turn"),
        },
    )


class TextAdaptive(AdaptiveScenario):
    """
    Adaptive text-attack scenario.

    Selects techniques per-objective via an epsilon-greedy selector over the
    set of selected strategies. ``prompt_sending`` participates as one of the
    selector's techniques rather than being prepended as a baseline.
    """

    VERSION: int = 1
    _atomic_attack_prefix: ClassVar[str] = "adaptive"
    _cached_strategy_class: ClassVar[type[ScenarioStrategy] | None] = None

    @classmethod
    def get_strategy_class(cls) -> type[ScenarioStrategy]:
        """Return the strategy enum for this scenario, building it once on first access."""
        if cls._cached_strategy_class is None:
            cls._cached_strategy_class = _build_text_adaptive_strategy()
        return cls._cached_strategy_class

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        """Return the default strategy aggregate (resolves to every ``default``-tagged technique)."""
        strategy_class = cls.get_strategy_class()
        return strategy_class("default")

    @classmethod
    def required_datasets(cls) -> list[str]:
        """Return the dataset names this scenario expects when no override is provided."""
        return [
            "airt_hate",
            "airt_fairness",
            "airt_violence",
            "airt_sexual",
            "airt_harassment",
            "airt_misinformation",
            "airt_leakage",
        ]

    @classmethod
    def default_dataset_config(cls) -> DatasetConfiguration:
        """Return the default :class:`DatasetConfiguration` (required datasets, capped at 4 per dataset)."""
        return DatasetConfiguration(dataset_names=cls.required_datasets(), max_dataset_size=4)

    @apply_defaults
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
                context key. Defaults to ``global_context``. Use
                ``harm_category_context`` to partition by harm category.
            scenario_result_id (str | None): ID of an existing ``ScenarioResult`` to resume.
        """
        super().__init__(
            objective_scorer=objective_scorer,
            epsilon=epsilon,
            pool_threshold=pool_threshold,
            max_attempts_per_objective=max_attempts_per_objective,
            seed=seed,
            context_extractor=context_extractor,
            scenario_result_id=scenario_result_id,
        )
