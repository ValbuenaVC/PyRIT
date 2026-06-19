# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
InspectTaskFactory — the single entry point for creating and running Inspect evals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrit.executor.attack.core.attack_strategy import AttackStrategy
    from pyrit.models.seeds.seed_attack_group import SeedAttackGroup
    from pyrit.prompt_target import PromptTarget


class InspectTaskFactory:
    """
    Single call site that assembles and runs Inspect evals from PyRIT components.

    The only bridge entry point that scenarios and scorers should use. Construction
    is cheap and synchronous; initialization is async and is awaited by the
    ``run_*_async`` methods (no async work in ``__init__``).
    """

    def __init__(self) -> None:
        """Initialize the InspectTaskFactory."""
        raise NotImplementedError

    def create_task(
        self,
        *,
        seed_groups: list[SeedAttackGroup],
        target: PromptTarget | None = None,
        attack: AttackStrategy | None = None,
        solver_name: str | None = None,
        inspect_scorer_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> Any:
        """
        Create an Inspect ``Task`` from PyRIT components.

        Synchronous. Assumes that ``InspectInitializer`` has already been run;
        raises a guided ``InspectBridgeError`` if not.

        Args:
            seed_groups (list[SeedAttackGroup]): Attack seed groups. Each must
                be a ``SeedAttackGroup`` with exactly one objective.
            target (PromptTarget | None): Optional PyRIT target. If ``None``,
                ``model`` must be passed to ``run_task_async``.
            attack (AttackStrategy | None): PyRIT attack to use as the solver.
                Provide ``attack`` XOR ``solver_name``.
            solver_name (str | None): Name of an Inspect-native solver (e.g. ``"react"``).
                Provide ``attack`` XOR ``solver_name``.
            inspect_scorer_name (str | None): Registered Inspect scorer for the eval.
            config (dict[str, Any] | None): Optional task configuration overrides.

        Returns:
            inspect_ai.Task: The assembled Inspect task.

        Raises:
            InspectBridgeError: If not initialized or if both/neither of
                ``attack`` and ``solver_name`` are provided.
            ValueError: If ``seed_groups`` is empty.

        """
        raise NotImplementedError

    async def run_task_async(self, task: Any, *, model: str | None = None, **eval_kwargs: Any) -> list[Any]:
        """
        Ensure initialized, then run an Inspect task via ``inspect_ai.eval``.

        Awaits ``InspectInitializer().initialize_async()`` (idempotent) before
        running the eval. Logs are persisted through the ``MemoryAdapter`` hook.

        Args:
            task: The Inspect ``Task`` to run (typically from ``create_task``).
            model (str | None): Optional Inspect model string (e.g. ``"pyrit/my_target"``).
            **eval_kwargs: Additional keyword arguments forwarded to ``inspect_ai.eval``.

        Returns:
            list[EvalLog]: Eval log entries from the run.

        """
        raise NotImplementedError

    async def run_named_eval_async(
        self,
        *,
        eval_name: str,
        target: PromptTarget | None = None,
        model: str | None = None,
        **eval_kwargs: Any,
    ) -> list[Any]:
        """
        Run an external named Inspect eval where PyRIT supplies only the model-under-test.

        Used for external benchmarks (e.g. ``"cybench"``) that bring their own
        task/agent/tools. PyRIT supplies the wrapped target as the model.

        Args:
            eval_name (str): Name of the registered Inspect eval/benchmark to run.
            target (PromptTarget | None): Optional PyRIT target to use as the
                model-under-test. If ``None``, ``model`` must be provided.
            model (str | None): Optional Inspect model string override.
            **eval_kwargs: Additional keyword arguments forwarded to ``inspect_ai.eval``.

        Returns:
            list[EvalLog]: Eval log entries from the run.

        """
        raise NotImplementedError
