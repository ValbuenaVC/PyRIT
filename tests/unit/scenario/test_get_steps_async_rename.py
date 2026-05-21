# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests pinning the R2 deprecation contract for the
``_get_atomic_attacks_async`` → ``_get_steps_async`` rename.

The rename ships with a passthrough shim plus an ``__init_subclass__``
deprecation hook on :class:`Scenario`. These tests pin:

* Subclasses overriding the legacy name only → ``DeprecationWarning`` fires at
  class creation, and the orchestrator still receives the legacy override's
  output via the base ``_get_steps_async`` delegation path.
* Subclasses overriding the new name only → no warning; legacy
  ``_get_atomic_attacks_async`` callers still get the new override's output via
  the base passthrough.
* Subclasses overriding both → no warning; new name wins.
* Subclasses overriding neither → no warning; default factory body runs.
"""

from __future__ import annotations

import warnings
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from pyrit.scenario.core.atomic_attack import AtomicAttack
from pyrit.scenario.core.dataset_configuration import DatasetConfiguration
from pyrit.scenario.core.scenario import BaselineAttackPolicy, Scenario
from pyrit.scenario.core.scenario_strategy import ScenarioStrategy


class _R2Strategy(ScenarioStrategy):
    DEFAULT = ("default", {"default"})


class _R2ScenarioBase(Scenario):
    """Common abstract-method satisfaction for the R2 rename tests."""

    BASELINE_ATTACK_POLICY: ClassVar[BaselineAttackPolicy] = BaselineAttackPolicy.Disabled

    @classmethod
    def get_strategy_class(cls):
        return _R2Strategy

    @classmethod
    def get_default_strategy(cls):
        return _R2Strategy.DEFAULT

    @classmethod
    def default_dataset_config(cls) -> DatasetConfiguration:
        return DatasetConfiguration()


def _build_scenario_kwargs() -> dict:
    objective_scorer = MagicMock()
    objective_scorer.get_identifier.return_value = {"id": "r2-scorer"}
    objective_scorer.get_scorer_metrics.return_value = None
    return {
        "name": "r2",
        "version": 1,
        "strategy_class": _R2Strategy,
        "objective_scorer": objective_scorer,
    }


class TestGetStepsAsyncRenameDeprecation:
    """Pin the class-creation deprecation warning surface."""

    def test_legacy_override_only_emits_deprecation(self) -> None:
        with pytest.warns(DeprecationWarning, match=r"_get_atomic_attacks_async"):

            class _LegacyOnly(_R2ScenarioBase):
                async def _get_atomic_attacks_async(self) -> list[AtomicAttack]:
                    return []

        assert _LegacyOnly is not None

    def test_new_override_only_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            class _NewOnly(_R2ScenarioBase):
                async def _get_steps_async(self) -> list[AtomicAttack]:
                    return []

        assert _NewOnly is not None

    def test_both_overrides_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            class _Both(_R2ScenarioBase):
                async def _get_steps_async(self) -> list[AtomicAttack]:
                    return []

                async def _get_atomic_attacks_async(self) -> list[AtomicAttack]:
                    return []

        assert _Both is not None

    def test_no_override_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            class _Neither(_R2ScenarioBase):
                pass

        assert _Neither is not None


class TestGetStepsAsyncRenameDelegation:
    """Pin the runtime delegation behavior of both directions."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("patch_central_database")
    async def test_legacy_override_reached_via_get_steps_async(self) -> None:
        sentinel: list[AtomicAttack] = [MagicMock(spec=AtomicAttack)]

        with pytest.warns(DeprecationWarning):

            class _LegacyOnly(_R2ScenarioBase):
                async def _get_atomic_attacks_async(self) -> list[AtomicAttack]:
                    return sentinel

        scenario = _LegacyOnly(**_build_scenario_kwargs())
        result = await scenario._get_steps_async()
        assert result is sentinel

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("patch_central_database")
    async def test_new_override_reached_via_legacy_passthrough(self) -> None:
        sentinel: list[AtomicAttack] = [MagicMock(spec=AtomicAttack)]

        class _NewOnly(_R2ScenarioBase):
            async def _get_steps_async(self) -> list[AtomicAttack]:
                return sentinel

        scenario = _NewOnly(**_build_scenario_kwargs())
        result = await scenario._get_atomic_attacks_async()
        assert result is sentinel
