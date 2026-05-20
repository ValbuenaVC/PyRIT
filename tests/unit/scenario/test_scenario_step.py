# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for ``pyrit.scenario.core.scenario_step``."""

import pytest

from pyrit.identifiers import ComponentIdentifier
from pyrit.scenario.core.scenario_step import ScenarioStep, ScenarioStepResult


class _ConcreteStep(ScenarioStep):
    """Minimal concrete step that returns a fixed outcome."""

    def __init__(self, *, name: str = "test_step", outputs: list[str] | None = None):
        self.name = name
        self.outputs = outputs or ["done"]

    async def process_async(self) -> ScenarioStepResult:
        return ScenarioStepResult(outcome="done")


def test_scenario_step_is_abstract():
    with pytest.raises(TypeError):
        # Cannot instantiate the ABC directly because ``process_async`` is abstract.
        ScenarioStep()  # type: ignore[abstract]


def test_concrete_step_can_be_instantiated():
    step = _ConcreteStep()
    assert step.name == "test_step"
    assert step.outputs == ["done"]


async def test_process_async_returns_step_result():
    step = _ConcreteStep()
    result = await step.process_async()
    assert isinstance(result, ScenarioStepResult)
    assert result.outcome == "done"
    assert result.attack_results == []
    assert result.step_identifier is None


def test_get_identifier_includes_name_and_outputs():
    step = _ConcreteStep(name="opening", outputs=["pass", "fail"])
    identifier = step.get_identifier()
    assert isinstance(identifier, ComponentIdentifier)
    assert identifier.params["name"] == "opening"
    assert identifier.params["outputs"] == ["pass", "fail"]
    assert identifier.class_name == "_ConcreteStep"


def test_get_identifier_is_cached():
    step = _ConcreteStep()
    first = step.get_identifier()
    second = step.get_identifier()
    assert first is second


def test_step_result_is_frozen():
    result = ScenarioStepResult(outcome="done")
    with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
        result.outcome = "other"  # type: ignore[misc]


def test_step_result_defaults():
    result = ScenarioStepResult(outcome="done")
    assert result.attack_results == []
    assert result.step_identifier is None
