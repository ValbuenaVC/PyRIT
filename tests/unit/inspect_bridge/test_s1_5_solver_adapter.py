# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Unit tests for S1.5 — AttackToSolverAdapter.

All inspect_ai imports are deferred to ensure we don't pollute ``import pyrit``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def _make_mock_attack(objective_text: str = "test objective") -> MagicMock:
    """Build a MagicMock with the AttackStrategy spec."""
    from pyrit.executor.attack.core.attack_strategy import AttackStrategy

    attack = MagicMock(spec=AttackStrategy)
    attack.execute_async = AsyncMock(return_value=MagicMock())
    return attack


def _make_task_state(user_text: str = "hello", target_text: str = "objective") -> object:
    """Build an Inspect TaskState for tests."""
    from inspect_ai.model import ChatMessageUser
    from inspect_ai.solver import TaskState

    return TaskState(
        model="pyrit/test",
        sample_id=1,
        epoch=1,
        input=user_text,
        messages=[ChatMessageUser(content=user_text)],
        target=target_text,  # type: ignore[arg-type]
    )


def test_solver_adapter_stores_attack() -> None:
    from pyrit.inspect_bridge import AttackToSolverAdapter

    attack = _make_mock_attack()
    adapter = AttackToSolverAdapter(attack=attack)
    assert adapter._attack is attack


def test_to_solver_returns_callable() -> None:
    from pyrit.inspect_bridge import AttackToSolverAdapter

    attack = _make_mock_attack()
    adapter = AttackToSolverAdapter(attack=attack)
    solver = adapter.to_solver()
    assert callable(solver)


async def test_solver_calls_execute_async() -> None:
    from pyrit.inspect_bridge import AttackToSolverAdapter

    attack = _make_mock_attack()
    adapter = AttackToSolverAdapter(attack=attack)
    solver = adapter.to_solver()
    state = _make_task_state(user_text="attack prompt", target_text="my objective")

    generate = MagicMock()
    await solver(state, generate)

    attack.execute_async.assert_awaited_once()
    call_kwargs = attack.execute_async.call_args.kwargs
    assert call_kwargs["objective"] == "my objective"


async def test_solver_passes_user_prompt_as_next_message() -> None:
    from pyrit.inspect_bridge import AttackToSolverAdapter
    from pyrit.models import Message

    attack = _make_mock_attack()
    adapter = AttackToSolverAdapter(attack=attack)
    solver = adapter.to_solver()
    state = _make_task_state(user_text="initial attack message", target_text="harm objective")

    generate = MagicMock()
    await solver(state, generate)

    call_kwargs = attack.execute_async.call_args.kwargs
    next_msg = call_kwargs.get("next_message")
    assert next_msg is not None
    assert isinstance(next_msg, Message)
    piece = next_msg.message_pieces[0]
    assert piece.original_value == "initial attack message"
    assert piece.role == "user"


async def test_solver_marks_state_completed() -> None:
    from pyrit.inspect_bridge import AttackToSolverAdapter

    attack = _make_mock_attack()
    adapter = AttackToSolverAdapter(attack=attack)
    solver = adapter.to_solver()
    state = _make_task_state()

    generate = MagicMock()
    result = await solver(state, generate)

    assert result.completed is True


async def test_solver_returns_task_state() -> None:
    from inspect_ai.solver import TaskState

    from pyrit.inspect_bridge import AttackToSolverAdapter

    attack = _make_mock_attack()
    adapter = AttackToSolverAdapter(attack=attack)
    solver = adapter.to_solver()
    state = _make_task_state()

    generate = MagicMock()
    result = await solver(state, generate)

    assert isinstance(result, TaskState)


async def test_solver_empty_target_uses_empty_objective() -> None:
    from pyrit.inspect_bridge import AttackToSolverAdapter

    attack = _make_mock_attack()
    adapter = AttackToSolverAdapter(attack=attack)
    solver = adapter.to_solver()
    state = _make_task_state(target_text="")

    generate = MagicMock()
    await solver(state, generate)

    call_kwargs = attack.execute_async.call_args.kwargs
    assert call_kwargs["objective"] == ""


def test_different_attacks_produce_different_solvers() -> None:
    from pyrit.inspect_bridge import AttackToSolverAdapter

    attack1 = _make_mock_attack()
    attack2 = _make_mock_attack()
    solver1 = AttackToSolverAdapter(attack=attack1).to_solver()
    solver2 = AttackToSolverAdapter(attack=attack2).to_solver()
    assert solver1 is not solver2
