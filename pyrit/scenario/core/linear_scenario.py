# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``LinearScenario`` — L0 authoring tier for "I just want to run these steps in order".

This module exists so scenario authors can hand a list of pre-built
``ScenarioStep`` instances (typically ``AtomicAttack`` objects) to a
zero-subclass scenario and get a runnable :class:`Scenario` back. Composes
purely on top of the default linear policy built by
:meth:`Scenario._build_default_linear_policy`; no graph or state vocabulary
is exposed to the caller.

L1 (override ``_get_atomic_attacks_async``) and L2 (override
``_build_execution_graph``) remain available for scenarios that need
registry-driven technique selection or non-linear control flow respectively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional, cast

from pyrit.scenario.core.dataset_configuration import DatasetConfiguration
from pyrit.scenario.core.scenario import BaselineAttackPolicy, Scenario
from pyrit.scenario.core.scenario_strategy import ScenarioStrategy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyrit.scenario.core.atomic_attack import AtomicAttack
    from pyrit.scenario.core.scenario_step import ScenarioStep
    from pyrit.score import Scorer


class LinearScenarioStrategy(ScenarioStrategy):
    """
    Single-member strategy enum for :class:`LinearScenario`.

    ``LinearScenario`` doesn't run a technique-selection menu — the steps
    are constructor inputs. The strategy enum exists only to satisfy the
    base ``Scenario`` contract.
    """

    DEFAULT = ("default", {"all"})

    @classmethod
    def get_aggregate_tags(cls) -> set[str]:
        """Return the strategy aggregate tags this enum exposes."""
        return {"all"}


class LinearScenario(Scenario):
    """
    Run a caller-supplied list of ``ScenarioStep`` instances in order.

    The L0 authoring tier for scenarios that don't need a custom subclass,
    a technique registry, or non-linear control flow. The supplied steps
    are walked by the default linear policy (:meth:`Scenario._build_default_linear_policy`),
    so ``AtomicAttack`` steps automatically receive scenario-level
    ``max_concurrency`` via :meth:`AtomicAttack.set_scenario_max_concurrency`
    and all dispatch flows uniformly through ``step.process_async``.

    Example:
        ```python
        scenario = LinearScenario(
            steps=[atomic_a, atomic_b],
            objective_scorer=my_scorer,
        )
        await scenario.initialize_async(objective_target=target)
        result = await scenario.run_async()
        ```
    """

    #: Default scenario version; bump if behavior changes in a way that
    #: invalidates resume from an older persisted ``ScenarioResult``.
    VERSION: ClassVar[int] = 1

    #: The steps are supplied directly; baseline injection would mutate
    #: that list and surprise the caller.
    BASELINE_ATTACK_POLICY: ClassVar[BaselineAttackPolicy] = BaselineAttackPolicy.Forbidden

    def __init__(
        self,
        *,
        steps: Sequence[ScenarioStep],
        objective_scorer: Scorer,
        name: str = "",
        version: int | None = None,
        scenario_result_id: Optional[str] = None,
    ) -> None:
        """
        Initialize a linear scenario from a caller-supplied step list.

        Args:
            steps (Sequence[ScenarioStep]): Steps to walk in order. Must be
                non-empty.
            objective_scorer (Scorer): Forwarded to the base ``Scenario``.
                Used by baseline / retry paths; for L0 usage with explicit
                steps and ``BASELINE_ATTACK_POLICY = Forbidden``, it is
                effectively only consulted if a custom retry path is wired in.
            name (str): Descriptive name for the scenario.
            version (int | None): Scenario version. Defaults to ``VERSION``;
                override only when callers want to invalidate resume from a
                prior shape.
            scenario_result_id (str | None): Optional ID of an existing
                scenario result to resume.

        Raises:
            ValueError: If ``steps`` is empty.
        """
        if not steps:
            raise ValueError("LinearScenario requires at least one step.")

        self._explicit_steps: list[ScenarioStep] = list(steps)

        super().__init__(
            name=name,
            version=version if version is not None else self.VERSION,
            strategy_class=self.get_strategy_class(),
            objective_scorer=objective_scorer,
            scenario_result_id=scenario_result_id,
        )

    @classmethod
    def get_strategy_class(cls) -> type[ScenarioStrategy]:
        """Return the single-member strategy enum class."""
        return LinearScenarioStrategy

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        """Return the only strategy member."""
        return LinearScenarioStrategy.DEFAULT

    @classmethod
    def default_dataset_config(cls) -> DatasetConfiguration:
        """
        Return an empty dataset configuration.

        Steps are supplied directly via the constructor, so the base
        scenario's auto-build from registered datasets is unused.

        Returns:
            DatasetConfiguration: An empty configuration.
        """
        return DatasetConfiguration()

    async def _get_atomic_attacks_async(self) -> list[AtomicAttack]:
        """
        Return the caller-supplied steps in order.

        The return type is :class:`list[AtomicAttack]` for parity with the
        base ``Scenario._get_atomic_attacks_async`` contract; the resume /
        orchestrator code reads the duck-typed attributes that every
        ``ScenarioStep`` exposes (``name``, ``process_async``) so non-
        ``AtomicAttack`` subclasses pass through cleanly.

        Returns:
            list[AtomicAttack]: The caller-supplied steps, cast to satisfy the
                base contract's type signature.
        """
        return cast("list[AtomicAttack]", list(self._explicit_steps))
