# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Unit tests for S1.7 — InspectTaskFactory.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch


def _make_seed_attack_group(prompt_text: str = "attack prompt", objective: str = "test objective") -> MagicMock:
    """Build a MagicMock with the SeedAttackGroup spec."""
    from pyrit.models.seeds.seed_attack_group import SeedAttackGroup

    group = MagicMock(spec=SeedAttackGroup)
    group.objective = MagicMock()
    group.objective.value = objective
    prompt = MagicMock()
    prompt.value = prompt_text
    group.prompts = [prompt]
    return group


def _make_mock_target() -> MagicMock:
    from pyrit.prompt_target import PromptTarget

    return MagicMock(spec=PromptTarget)


def _make_mock_attack() -> MagicMock:
    from pyrit.executor.attack.core.attack_strategy import AttackStrategy

    attack = MagicMock(spec=AttackStrategy)
    attack.execute_async = AsyncMock(return_value=MagicMock())
    return attack


def test_task_factory_constructs() -> None:
    from pyrit.inspect_bridge import InspectTaskFactory

    factory = InspectTaskFactory()
    assert factory is not None


def test_create_task_returns_inspect_task() -> None:
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample

    from pyrit.inspect_bridge import InspectTaskFactory

    factory = InspectTaskFactory()
    group = _make_seed_attack_group()
    attack = _make_mock_attack()
    mock_solver = MagicMock()
    mock_dataset = MemoryDataset(samples=[Sample(input="test prompt", target="test objective")])

    with (
        patch("pyrit.inspect_bridge._task_factory._initialized", True),
        patch("pyrit.inspect_bridge._task_factory.DatasetAdapter") as mock_ds_cls,
        patch("pyrit.inspect_bridge._task_factory.AttackToSolverAdapter") as mock_solver_cls,
    ):
        mock_ds_cls.return_value.to_inspect_dataset.return_value = mock_dataset
        mock_solver_cls.return_value.to_solver.return_value = mock_solver
        task = factory.create_task(seed_groups=[group], attack=attack)

    assert isinstance(task, Task)


def test_create_task_raises_if_not_initialized() -> None:
    from pyrit.inspect_bridge import InspectBridgeError, InspectTaskFactory

    factory = InspectTaskFactory()
    group = _make_seed_attack_group()

    with patch("pyrit.inspect_bridge._task_factory._initialized", False):
        try:
            factory.create_task(seed_groups=[group], solver_name="generate")
            raise AssertionError("Should have raised InspectBridgeError")
        except InspectBridgeError as e:
            assert "InspectInitializer" in str(e)


def test_create_task_raises_if_both_attack_and_solver_name() -> None:
    from pyrit.inspect_bridge import InspectBridgeError, InspectTaskFactory

    factory = InspectTaskFactory()
    group = _make_seed_attack_group()
    attack = _make_mock_attack()

    with patch("pyrit.inspect_bridge._task_factory._initialized", True):
        try:
            factory.create_task(seed_groups=[group], attack=attack, solver_name="generate")
            raise AssertionError("Should have raised InspectBridgeError")
        except InspectBridgeError:
            pass


def test_create_task_raises_if_neither_attack_nor_solver_name() -> None:
    from pyrit.inspect_bridge import InspectBridgeError, InspectTaskFactory

    factory = InspectTaskFactory()
    group = _make_seed_attack_group()

    with patch("pyrit.inspect_bridge._task_factory._initialized", True):
        try:
            factory.create_task(seed_groups=[group])
            raise AssertionError("Should have raised InspectBridgeError")
        except InspectBridgeError:
            pass


def test_create_task_raises_on_empty_seed_groups() -> None:
    from pyrit.inspect_bridge import InspectTaskFactory

    factory = InspectTaskFactory()

    with patch("pyrit.inspect_bridge._task_factory._initialized", True):
        try:
            factory.create_task(seed_groups=[], attack=_make_mock_attack())
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass


async def test_run_task_async_calls_eval_async() -> None:
    from pyrit.inspect_bridge import InspectTaskFactory

    factory = InspectTaskFactory()
    mock_task = MagicMock()
    mock_logs = [MagicMock()]

    with (
        patch("pyrit.inspect_bridge._task_factory.InspectInitializer") as mock_init_cls,
        patch("inspect_ai.eval_async", new_callable=AsyncMock, return_value=mock_logs),
    ):
        mock_init_cls.return_value.initialize_async = AsyncMock()
        result = await factory.run_task_async(mock_task, model="pyrit/test")

    assert result == mock_logs


async def test_run_named_eval_async_uses_target_model_name() -> None:
    from pyrit.inspect_bridge import InspectTaskFactory

    factory = InspectTaskFactory()
    target = _make_mock_target()
    target.get_identifier.return_value.unique_name = "my_target"
    mock_logs = [MagicMock()]

    with (
        patch("pyrit.inspect_bridge._task_factory.InspectInitializer") as mock_init_cls,
        patch("inspect_ai.eval_async", new_callable=AsyncMock, return_value=mock_logs),
    ):
        mock_init_cls.return_value.initialize_async = AsyncMock()
        result = await factory.run_named_eval_async(eval_name="cybench", target=target)

    assert result == mock_logs
