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
        result.outcome = "other"  # type: ignore[ty:invalid-assignment]


def test_step_result_defaults():
    result = ScenarioStepResult(outcome="done")
    assert result.attack_results == []
    assert result.step_identifier is None


def test_step_result_outcome_is_required():
    with pytest.raises(TypeError):
        ScenarioStepResult()  # type: ignore[call-arg]


def test_step_result_metadata_defaults_to_fresh_dict_per_instance():
    first = ScenarioStepResult(outcome="done")
    second = ScenarioStepResult(outcome="done")
    assert first.metadata == {}
    assert first.metadata is not second.metadata
    first.metadata["k"] = "v"
    assert second.metadata == {}


def test_step_result_attack_results_defaults_to_fresh_list_per_instance():
    first = ScenarioStepResult(outcome="done")
    second = ScenarioStepResult(outcome="done")
    assert first.attack_results == []
    assert first.attack_results is not second.attack_results
    first.attack_results.append("sentinel")  # type: ignore[arg-type]
    assert second.attack_results == []


def test_step_result_accepts_all_fields():
    identifier = ComponentIdentifier.of(_ConcreteStep(), params={"name": "x", "outputs": ["done"]})
    metadata = {"step_name": "x", "extra": 1}
    result = ScenarioStepResult(
        outcome="success",
        attack_results=[],
        step_identifier=identifier,
        metadata=metadata,
    )
    assert result.outcome == "success"
    assert result.step_identifier is identifier
    assert result.metadata == metadata


class _StepWithoutProcessAsync(ScenarioStep):
    """Subclass that forgets to implement ``process_async``."""

    def __init__(self) -> None:
        self.name = "incomplete"
        self.outputs = ["done"]


def test_subclass_missing_process_async_cannot_instantiate():
    with pytest.raises(TypeError, match="process_async"):
        _StepWithoutProcessAsync()  # type: ignore[abstract]


class _StepWithDefaultIdentifier(ScenarioStep):
    """Subclass that overrides only ``process_async`` — inherits identifier behavior."""

    def __init__(self) -> None:
        self.name = "inherits_identifier"
        self.outputs = ["done", "skipped"]

    async def process_async(self) -> ScenarioStepResult:
        return ScenarioStepResult(outcome="done")


def test_subclass_inherits_default_build_identifier():
    step = _StepWithDefaultIdentifier()
    identifier = step.get_identifier()
    assert identifier.params["name"] == "inherits_identifier"
    assert identifier.params["outputs"] == ["done", "skipped"]
    assert identifier.children == {}
