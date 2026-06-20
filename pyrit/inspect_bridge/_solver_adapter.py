# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
AttackToSolverAdapter — wraps a PyRIT AttackStrategy as an Inspect AI Solver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrit.executor.attack.core.attack_strategy import AttackStrategy


class AttackToSolverAdapter:
    """
    Wraps a PyRIT ``AttackStrategy`` so Inspect AI can run it as a ``Solver``.

    The returned solver drives the PyRIT attack against the sample's objective
    and writes results into the Inspect ``TaskState``.
    """

    def __init__(self, *, attack: AttackStrategy) -> None:
        """
        Initialize the AttackToSolverAdapter.

        Args:
            attack (AttackStrategy): The PyRIT attack strategy to wrap.

        """
        self._attack = attack

    def to_solver(self) -> Any:
        """
        Return an Inspect AI ``Solver`` that drives the wrapped attack.

        Inspect calls the returned solver for each sample, passing the
        current ``TaskState`` and a ``Generate`` callable. The solver
        extracts the objective and initial user message from the state,
        runs the PyRIT attack, marks the state as completed, and returns it.

        Returns:
            inspect_ai.solver.Solver: A solver that runs the PyRIT attack
            against each sample's objective.

        """
        from pyrit.models import Message, MessagePiece

        attack = self._attack

        # @solver decorates a *factory* that returns an async callable (Solver),
        # not the async callable itself. We use the two-level pattern: a factory
        # decorated with @solver that returns the inner async function.
        async def _run(state: Any, generate: Any) -> Any:  # pyrit-async-suffix-exempt
            target = getattr(state, "target", None)
            # target can be a Target object (has .text) or a plain string
            if hasattr(target, "text"):
                objective: str = target.text
            else:
                objective = str(target) if target is not None else ""

            user_prompt_msg = getattr(state, "user_prompt", None)
            if hasattr(user_prompt_msg, "text"):
                prompt_text: str = user_prompt_msg.text
            else:
                prompt_text = str(user_prompt_msg) if user_prompt_msg is not None else ""

            piece = MessagePiece(role="user", original_value=prompt_text)
            next_message = Message(message_pieces=[piece])

            await attack.execute_async(objective=objective, next_message=next_message)
            state.completed = True
            return state

        return _run
