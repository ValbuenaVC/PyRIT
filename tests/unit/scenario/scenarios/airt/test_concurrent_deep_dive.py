# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
R4 — concurrent deep-dive dispatch in ``FilteredDeepDiveStep``.

Pins the contract that adding ``max_step_concurrency > 1`` to
``FilteredDeepDiveStep`` (and the ``BroadSweepThenDeepDive`` scenario that
plumbs it) fans out the per-atomic dispatch through ``asyncio.gather`` under
a semaphore while preserving:

* input order of ``dispatched_categories`` and ``attack_results``;
* short-circuit on empty weak-set;
* bit-for-bit semantics at the default ``max_step_concurrency=1``;
* fail-fast validation on bogus values.

The hardest contract to pin is "multiple atomics are genuinely in-flight at
the same time". That is asserted via an ``asyncio.Event``-gated fake
``AtomicAttack.run_async`` that records the peak observed in-flight count
and only releases once the cap is reached. With ``max_step_concurrency=N``
and ``N`` candidates, all ``N`` must reach the gate before any completes.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, PropertyMock

import pytest

from pyrit.executor.attack.core import AttackExecutorResult
from pyrit.identifiers import ComponentIdentifier
from pyrit.models import AttackOutcome, AttackResult
from pyrit.scenario.core import AtomicAttack
from pyrit.scenario.scenarios.airt.sweep_then_deep_dive import (
    FilteredDeepDiveStep,
)


def _attack_result(*, conversation_id: str, objective: str) -> AttackResult:
    result = AttackResult(
        conversation_id=conversation_id,
        objective=objective,
        outcome=AttackOutcome.SUCCESS,
        executed_turns=1,
    )
    result.atomic_attack_identifier = ComponentIdentifier(
        class_name="MockAttack",
        class_module="tests.unit.scenario.scenarios.airt",
        params={"name": conversation_id},
    )
    return result


def _make_atomic_mock(
    *,
    name: str,
    display_group: str,
    attack_results: list[AttackResult],
) -> MagicMock:
    attack = MagicMock(spec=AtomicAttack)
    attack.atomic_attack_name = name
    attack.display_group = display_group
    type(attack).objectives = PropertyMock(return_value=[r.objective for r in attack_results])
    type(attack).seed_groups = PropertyMock(return_value=[])
    attack.get_identifier.return_value = ComponentIdentifier(
        class_name="AtomicAttack",
        class_module="tests.unit.scenario.scenarios.airt",
        params={"name": name},
    )

    async def _fake_run(*args: Any, **kwargs: Any) -> AttackExecutorResult:
        return AttackExecutorResult(completed_results=attack_results, incomplete_objectives=[])

    attack.run_async = MagicMock(side_effect=_fake_run)
    return attack


def _make_gated_atomic_mock(
    *,
    name: str,
    display_group: str,
    attack_results: list[AttackResult],
    inflight_counter: dict[str, int],
    peak: dict[str, int],
    release_event: asyncio.Event,
) -> MagicMock:
    """Build an AtomicAttack mock whose run_async blocks on ``release_event``.

    Tracks how many invocations are simultaneously past the gate (in-flight)
    so the test can assert the peak concurrency reached.
    """
    attack = MagicMock(spec=AtomicAttack)
    attack.atomic_attack_name = name
    attack.display_group = display_group
    type(attack).objectives = PropertyMock(return_value=[r.objective for r in attack_results])
    type(attack).seed_groups = PropertyMock(return_value=[])
    attack.get_identifier.return_value = ComponentIdentifier(
        class_name="AtomicAttack",
        class_module="tests.unit.scenario.scenarios.airt",
        params={"name": name},
    )

    async def _gated_run(*args: Any, **kwargs: Any) -> AttackExecutorResult:
        inflight_counter["n"] += 1
        peak["n"] = max(peak["n"], inflight_counter["n"])
        try:
            await release_event.wait()
        finally:
            inflight_counter["n"] -= 1
        return AttackExecutorResult(completed_results=attack_results, incomplete_objectives=[])

    attack.run_async = MagicMock(side_effect=_gated_run)
    return attack


@pytest.mark.usefixtures("patch_central_database")
class TestFilteredDeepDiveStepConcurrency:
    """R4: deep-dive step fans out under asyncio.Semaphore when concurrency > 1."""

    def test_init_rejects_zero_max_step_concurrency(self) -> None:
        a = _make_atomic_mock(name="a", display_group="cat-a", attack_results=[])
        with pytest.raises(ValueError, match=r"max_step_concurrency must be >= 1"):
            FilteredDeepDiveStep(
                atomic_attack_name="deep",
                atomic_attacks=[a],
                weak_categories_ref=lambda: {"cat-a"},
                max_step_concurrency=0,
            )

    def test_init_rejects_negative_max_step_concurrency(self) -> None:
        a = _make_atomic_mock(name="a", display_group="cat-a", attack_results=[])
        with pytest.raises(ValueError, match=r"max_step_concurrency must be >= 1"):
            FilteredDeepDiveStep(
                atomic_attack_name="deep",
                atomic_attacks=[a],
                weak_categories_ref=lambda: {"cat-a"},
                max_step_concurrency=-3,
            )

    def test_default_concurrency_is_one(self) -> None:
        a = _make_atomic_mock(name="a", display_group="cat-a", attack_results=[])
        step = FilteredDeepDiveStep(
            atomic_attack_name="deep",
            atomic_attacks=[a],
            weak_categories_ref=lambda: {"cat-a"},
        )
        assert step._max_step_concurrency == 1

    async def test_dispatch_order_preserved_with_concurrency_gt_one(self) -> None:
        result_a = _attack_result(conversation_id="da", objective="oa")
        result_b = _attack_result(conversation_id="db", objective="ob")
        result_c = _attack_result(conversation_id="dc", objective="oc")
        atomic_a = _make_atomic_mock(name="a", display_group="cat-a", attack_results=[result_a])
        atomic_b = _make_atomic_mock(name="b", display_group="cat-b", attack_results=[result_b])
        atomic_c = _make_atomic_mock(name="c", display_group="cat-c", attack_results=[result_c])

        step = FilteredDeepDiveStep(
            atomic_attack_name="deep",
            atomic_attacks=[atomic_a, atomic_b, atomic_c],
            weak_categories_ref=lambda: {"cat-a", "cat-b", "cat-c"},
            max_step_concurrency=3,
        )

        step_result = await step.process_async()

        assert step_result.metadata["dispatched_categories"] == ["cat-a", "cat-b", "cat-c"]
        assert step_result.metadata["skipped_categories"] == []
        assert step_result.attack_results == [result_a, result_b, result_c]
        assert step_result.metadata["max_step_concurrency"] == 3

    async def test_skipped_categories_retain_input_order(self) -> None:
        result_b = _attack_result(conversation_id="db", objective="ob")
        atomic_a = _make_atomic_mock(name="a", display_group="cat-a", attack_results=[])
        atomic_b = _make_atomic_mock(name="b", display_group="cat-b", attack_results=[result_b])
        atomic_c = _make_atomic_mock(name="c", display_group="cat-c", attack_results=[])

        step = FilteredDeepDiveStep(
            atomic_attack_name="deep",
            atomic_attacks=[atomic_a, atomic_b, atomic_c],
            weak_categories_ref=lambda: {"cat-b"},
            max_step_concurrency=2,
        )

        step_result = await step.process_async()

        atomic_a.run_async.assert_not_called()
        atomic_b.run_async.assert_called_once()
        atomic_c.run_async.assert_not_called()
        assert step_result.metadata["dispatched_categories"] == ["cat-b"]
        assert step_result.metadata["skipped_categories"] == ["cat-a", "cat-c"]
        assert step_result.attack_results == [result_b]

    async def test_empty_weak_set_short_circuits_without_dispatch(self) -> None:
        atomic_a = _make_atomic_mock(name="a", display_group="cat-a", attack_results=[])
        atomic_b = _make_atomic_mock(name="b", display_group="cat-b", attack_results=[])

        step = FilteredDeepDiveStep(
            atomic_attack_name="deep",
            atomic_attacks=[atomic_a, atomic_b],
            weak_categories_ref=lambda: set(),
            max_step_concurrency=4,
        )

        step_result = await step.process_async()

        atomic_a.run_async.assert_not_called()
        atomic_b.run_async.assert_not_called()
        assert step_result.metadata["dispatched_categories"] == []
        assert step_result.metadata["skipped_categories"] == ["cat-a", "cat-b"]
        assert step_result.attack_results == []
        assert step_result.metadata["max_step_concurrency"] == 4

    async def test_concurrent_dispatch_observes_n_simultaneously_inflight(self) -> None:
        """All N eligible atomics must reach the gate before any completes when concurrency >= N."""
        inflight: dict[str, int] = {"n": 0}
        peak: dict[str, int] = {"n": 0}
        release = asyncio.Event()

        atomics = [
            _make_gated_atomic_mock(
                name=f"a{i}",
                display_group=f"cat-{i}",
                attack_results=[_attack_result(conversation_id=f"d{i}", objective=f"o{i}")],
                inflight_counter=inflight,
                peak=peak,
                release_event=release,
            )
            for i in range(4)
        ]

        step = FilteredDeepDiveStep(
            atomic_attack_name="deep",
            atomic_attacks=atomics,
            weak_categories_ref=lambda: {f"cat-{i}" for i in range(4)},
            max_step_concurrency=4,
        )

        gather_task = asyncio.create_task(step.process_async())
        # Yield until all 4 atomics have reached the gate.
        for _ in range(200):
            if inflight["n"] >= 4:
                break
            await asyncio.sleep(0)
        assert inflight["n"] == 4, f"expected 4 in-flight, got {inflight['n']}"
        release.set()
        result = await gather_task

        assert peak["n"] == 4
        assert result.metadata["max_step_concurrency"] == 4
        assert result.metadata["dispatched_categories"] == [f"cat-{i}" for i in range(4)]

    async def test_semaphore_bounds_inflight_below_candidate_count(self) -> None:
        """With ``max_step_concurrency=2`` and 4 candidates, peak in-flight must be <= 2."""
        inflight: dict[str, int] = {"n": 0}
        peak: dict[str, int] = {"n": 0}
        release = asyncio.Event()

        atomics = [
            _make_gated_atomic_mock(
                name=f"a{i}",
                display_group=f"cat-{i}",
                attack_results=[_attack_result(conversation_id=f"d{i}", objective=f"o{i}")],
                inflight_counter=inflight,
                peak=peak,
                release_event=release,
            )
            for i in range(4)
        ]

        step = FilteredDeepDiveStep(
            atomic_attack_name="deep",
            atomic_attacks=atomics,
            weak_categories_ref=lambda: {f"cat-{i}" for i in range(4)},
            max_step_concurrency=2,
        )

        gather_task = asyncio.create_task(step.process_async())
        # Yield long enough for the semaphore to have admitted 2 — and only 2.
        for _ in range(200):
            if inflight["n"] >= 2:
                break
            await asyncio.sleep(0)
        # Spin a few more iterations to confirm the cap holds.
        for _ in range(50):
            await asyncio.sleep(0)

        assert inflight["n"] == 2
        assert peak["n"] == 2
        release.set()
        result = await gather_task
        # After release, every atomic completed.
        assert len(result.attack_results) == 4

    async def test_default_concurrency_one_is_strictly_sequential(self) -> None:
        """With default ``max_step_concurrency=1``, only one atomic may be in-flight at a time."""
        inflight: dict[str, int] = {"n": 0}
        peak: dict[str, int] = {"n": 0}
        # Use a per-call release pattern: release immediately so the sequential
        # case still completes. Peak should never exceed 1.
        release = asyncio.Event()
        release.set()

        atomics = [
            _make_gated_atomic_mock(
                name=f"a{i}",
                display_group=f"cat-{i}",
                attack_results=[_attack_result(conversation_id=f"d{i}", objective=f"o{i}")],
                inflight_counter=inflight,
                peak=peak,
                release_event=release,
            )
            for i in range(3)
        ]

        step = FilteredDeepDiveStep(
            atomic_attack_name="deep",
            atomic_attacks=atomics,
            weak_categories_ref=lambda: {f"cat-{i}" for i in range(3)},
            # default max_step_concurrency=1
        )

        result = await step.process_async()
        assert peak["n"] == 1
        assert result.metadata["max_step_concurrency"] == 1
        assert len(result.attack_results) == 3


@pytest.mark.usefixtures("patch_central_database")
class TestBroadSweepThenDeepDiveConcurrencyPlumbing:
    """The scenario constructor must validate + forward ``max_step_concurrency``."""

    def test_scenario_rejects_zero_max_step_concurrency(self) -> None:
        from pyrit.scenario.scenarios.airt.sweep_then_deep_dive import (
            BroadSweepThenDeepDive,
        )

        atomic = _make_atomic_mock(name="a", display_group="cat-a", attack_results=[])
        scorer = MagicMock()
        scorer.outcomes = {"safety_violation", "safe"}

        with pytest.raises(ValueError, match=r"max_step_concurrency must be >= 1"):
            BroadSweepThenDeepDive(
                sweep_atomic_attack=atomic,
                deep_dive_atomic_attacks=[atomic],
                outcome_scorer=scorer,
                max_step_concurrency=0,
            )

    def test_scenario_rejects_negative_max_step_concurrency(self) -> None:
        from pyrit.scenario.scenarios.airt.sweep_then_deep_dive import (
            BroadSweepThenDeepDive,
        )

        atomic = _make_atomic_mock(name="a", display_group="cat-a", attack_results=[])
        scorer = MagicMock()
        scorer.outcomes = {"safety_violation", "safe"}

        with pytest.raises(ValueError, match=r"max_step_concurrency must be >= 1"):
            BroadSweepThenDeepDive(
                sweep_atomic_attack=atomic,
                deep_dive_atomic_attacks=[atomic],
                outcome_scorer=scorer,
                max_step_concurrency=-1,
            )

    def test_scenario_default_max_step_concurrency_is_one(self) -> None:
        from pyrit.scenario.scenarios.airt.sweep_then_deep_dive import (
            BroadSweepThenDeepDive,
        )

        atomic = _make_atomic_mock(name="a", display_group="cat-a", attack_results=[])
        scorer = MagicMock()
        scorer.outcomes = {"safety_violation", "safe"}

        scenario = BroadSweepThenDeepDive(
            sweep_atomic_attack=atomic,
            deep_dive_atomic_attacks=[atomic],
            outcome_scorer=scorer,
        )
        assert scenario._max_step_concurrency == 1

    def test_input_schema_advertises_max_step_concurrency_scalar(self) -> None:
        from pyrit.scenario.core.input_schema import RoleTag
        from pyrit.scenario.scenarios.airt.sweep_then_deep_dive import (
            BroadSweepThenDeepDive,
        )

        schema = BroadSweepThenDeepDive.input_schema()
        names = [r.name for r in schema]
        assert "max_step_concurrency" in names
        role = next(r for r in schema if r.name == "max_step_concurrency")
        assert role.tag is RoleTag.SCALAR
        assert role.param_type is int
        assert role.default == 1
        assert role.required is False
