# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``ScenarioStep`` — one outcome decision in a scenario's state machine.

A ``ScenarioStep`` is strictly coarser than an ``AtomicAttack``: it may invoke
multiple attack-technique executions whose collective results determine a
single transition label emitted to the surrounding ``StrategyGraph``'s
policy.

This module is part of the scenario core refactor scaffold (Phase 0). The ABC
is in place but no scenario consumes it yet; ``LinearAtomicStep`` (Phase 2)
becomes the first concrete implementation and ``AtomicAttack`` is then
re-rooted onto it for backward compatibility.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyrit.identifiers import ComponentIdentifier, Identifiable

if TYPE_CHECKING:
    from pyrit.models import AttackResult


@dataclass(frozen=True)
class ScenarioStepResult:
    """
    Outcome of a single ``ScenarioStep.process_async`` invocation.

    Attributes:
        outcome (str): The transition label emitted to the surrounding
            ``StrategyGraph``'s policy. Must be one of the step's declared
            ``outputs``.
        attack_results (list[AttackResult]): Every ``AttackResult`` produced
            during the step's execution, in the order they were produced.
            May be empty for steps that record outcomes from external signals
            rather than running attacks.
        step_identifier (ComponentIdentifier | None): The composite identifier
            for this step execution. ``None`` until Phase 4 lands the
            ``step_identifier`` persistence column; populated after that for
            scenarios that opt into graph-based execution.
    """

    outcome: str
    attack_results: list[AttackResult] = field(default_factory=list)
    step_identifier: ComponentIdentifier | None = None


class ScenarioStep(Identifiable):
    """
    Abstract base for one outcome decision in a scenario.

    Subclasses declare their valid input and output vocabulary, then implement
    ``process_async`` to execute attacks (or any other work) and return a
    ``ScenarioStepResult`` whose ``outcome`` is the transition label the
    surrounding ``StrategyGraph`` will use to advance state.

    Step granularity rule: one step owns *one* outcome decision. A step may
    wrap multiple attack-technique executions whose collective results
    determine that one transition label.
    """

    #: Display / resume key. Must be unique within an execution graph.
    name: str

    #: Declared transition labels this step can emit. ``"unscored"`` is
    #: implicitly always allowed for steps that use an ``OutcomeScorer``.
    outputs: list[str]

    @abstractmethod
    async def process_async(self) -> ScenarioStepResult:
        """
        Execute the step's work and return the outcome plus any attack results.

        Returns:
            ScenarioStepResult: The transition label and any attack results
                produced. The ``outcome`` must be one of ``self.outputs`` (or
                the implicit ``"unscored"`` sentinel from ``OutcomeScorer``).
        """
        ...

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build the behavioral identity for this step.

        Default implementation captures the step name and declared outputs.
        Subclasses should override to include their inputs (e.g., attack
        technique identifiers) and any other behavioral params.

        Returns:
            ComponentIdentifier: The frozen identity snapshot.
        """
        return ComponentIdentifier.of(
            self,
            params={
                "name": self.name,
                "outputs": list(self.outputs),
            },
        )
