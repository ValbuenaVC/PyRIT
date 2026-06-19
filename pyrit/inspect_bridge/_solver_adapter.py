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
        raise NotImplementedError

    def to_solver(self) -> Any:
        """
        Return an Inspect AI ``Solver`` that drives the wrapped attack.

        Returns:
            inspect_ai.solver.Solver: A solver that runs the PyRIT attack
            against each sample's objective.

        """
        raise NotImplementedError
