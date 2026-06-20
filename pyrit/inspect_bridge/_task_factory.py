# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
InspectTaskFactory — the single entry point for creating and running Inspect evals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyrit.inspect_bridge._dataset_adapter import DatasetAdapter
from pyrit.inspect_bridge._initializer import InspectInitializer, _initialized
from pyrit.inspect_bridge._solver_adapter import AttackToSolverAdapter

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
        from inspect_ai import Task
        from inspect_ai._util.registry import registry_lookup

        from pyrit.inspect_bridge.errors import InspectBridgeError

        if not _initialized:
            raise InspectBridgeError(
                message=(
                    "InspectInitializer has not been run. "
                    "Call await InspectInitializer().initialize_async() before create_task()."
                )
            )

        if not seed_groups:
            raise ValueError("seed_groups must not be empty")

        both_set = attack is not None and solver_name is not None
        neither_set = attack is None and solver_name is None
        if both_set or neither_set:
            raise InspectBridgeError(
                message="Provide attack XOR solver_name — exactly one must be specified."
            )

        dataset = DatasetAdapter(seed_groups=list(seed_groups)).to_inspect_dataset()

        if attack is not None:
            solver = AttackToSolverAdapter(attack=attack).to_solver()
        else:
            # solver_name is a registered Inspect solver name
            solver_fn = registry_lookup("solver", solver_name)
            if solver_fn is None:
                raise InspectBridgeError(message=f"Inspect solver '{solver_name}' not found in registry.")
            solver = solver_fn()

        scorer = None
        if inspect_scorer_name is not None:
            scorer_fn = registry_lookup("scorer", inspect_scorer_name)
            if scorer_fn is None:
                raise InspectBridgeError(message=f"Inspect scorer '{inspect_scorer_name}' not found in registry.")
            scorer = scorer_fn()

        model_arg = None
        if target is not None:
            from pyrit.inspect_bridge._target_adapter import TargetToModelAdapter

            model_arg = TargetToModelAdapter.model_name_for(target=target)

        task_config = config or {}
        return Task(
            dataset=dataset,
            solver=solver,
            scorer=scorer,
            model=model_arg,
            **task_config,
        )

    async def run_task_async(self, task: Any, *, model: str | None = None, **eval_kwargs: Any) -> list[Any]:
        """
        Ensure initialized, then run an Inspect task via ``inspect_ai.eval_async``.

        Awaits ``InspectInitializer().initialize_async()`` (idempotent) before
        running the eval. Logs are persisted through the ``MemoryAdapter`` hook.

        Args:
            task: The Inspect ``Task`` to run (typically from ``create_task``).
            model (str | None): Optional Inspect model string (e.g. ``"pyrit/my_target"``).
            **eval_kwargs: Additional keyword arguments forwarded to ``inspect_ai.eval_async``.

        Returns:
            list[EvalLog]: Eval log entries from the run.

        """
        from inspect_ai import eval_async

        await InspectInitializer().initialize_async()
        return await eval_async(task, model=model, **eval_kwargs)

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
            **eval_kwargs: Additional keyword arguments forwarded to ``inspect_ai.eval_async``.

        Returns:
            list[EvalLog]: Eval log entries from the run.

        """
        from inspect_ai import eval_async

        await InspectInitializer().initialize_async()

        resolved_model = model
        if resolved_model is None and target is not None:
            from pyrit.inspect_bridge._target_adapter import TargetToModelAdapter

            resolved_model = TargetToModelAdapter.model_name_for(target=target)

        return await eval_async(eval_name, model=resolved_model, **eval_kwargs)
