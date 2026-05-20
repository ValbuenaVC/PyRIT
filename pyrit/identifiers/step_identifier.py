# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario step identity builder.

Builds a composite ``ComponentIdentifier`` that wraps one or more
``atomic_attack_identifier``s with the step's name and emitted outcome label.
This is the *step-level* identity that the scenario state machine produces;
the underlying ``atomic_attack_identifier`` is **not** renamed or removed and
continues to live on every ``AttackResult`` exactly as before.

Bump ``STEP_EVAL_VERSION`` when changing what is included in the step
identifier; this lets ``StepEvaluationIdentifier`` (Phase 4) detect schema
drift without falsely conflating old and new hashes.

Composite shape::

    ScenarioStep
      ├── step_name          (param)
      ├── outcome            (param)
      ├── eval_version       (param)
      └── attack_executions  (children, list of atomic_attack_identifier)
"""

from __future__ import annotations

import logging
from typing import Any

from pyrit.identifiers.component_identifier import ComponentIdentifier

logger = logging.getLogger(__name__)

#: Schema version for ``build_step_identifier``. Bump on any change that
#: affects which params or children participate in the identity (and thus the
#: derived hash / eval hash). Stays inside the identifier's ``params`` so old
#: rows preserve their original version.
STEP_EVAL_VERSION: int = 1

_SCENARIO_STEP_CLASS_NAME = "ScenarioStep"
_SCENARIO_STEP_CLASS_MODULE = "pyrit.scenario.core.scenario_step"


def build_step_identifier(
    *,
    step_name: str,
    outcome: str,
    attack_execution_identifiers: list[ComponentIdentifier],
) -> ComponentIdentifier:
    """
    Build a composite ``ComponentIdentifier`` for one step execution.

    Args:
        step_name (str): The step's ``name`` attribute.
        outcome (str): The transition label the step emitted.
        attack_execution_identifiers (list[ComponentIdentifier]): The
            ``atomic_attack_identifier``s produced by every attack execution
            the step ran, in execution order. May be empty for steps that
            record outcomes from external signals.

            .. note::
               This list is stored as a ``list`` (not a set) and the resulting
               identifier hash IS order-sensitive: ``[A, B]`` and ``[B, A]``
               produce different identifiers. This is intentional — for
               adaptive / branching steps, the *trajectory* of attack
               attempts is a meaningful part of the step's run-time identity
               (the same techniques applied in different orders are different
               experiments). Callers that want order-invariant identity
               (e.g. "set of attacks tried, regardless of order") must sort
               the list themselves before passing it in.

    Returns:
        ComponentIdentifier: Composite identifier with
            ``class_name="ScenarioStep"`` and the attack executions nested
            under ``children["attack_executions"]``.

    Raises:
        ValueError: If ``step_name`` or ``outcome`` is empty.
    """
    if not step_name:
        raise ValueError("step_name must be non-empty.")
    if not outcome:
        raise ValueError("outcome must be non-empty.")

    params: dict[str, Any] = {
        "step_name": step_name,
        "outcome": outcome,
        "eval_version": STEP_EVAL_VERSION,
    }

    children: dict[str, Any] = {
        "attack_executions": list(attack_execution_identifiers),
    }

    return ComponentIdentifier(
        class_name=_SCENARIO_STEP_CLASS_NAME,
        class_module=_SCENARIO_STEP_CLASS_MODULE,
        params=params,
        children=children,
    )
