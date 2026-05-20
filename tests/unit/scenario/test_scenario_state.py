# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for ``pyrit.scenario.core.scenario_state``."""

from enum import Enum

from pyrit.scenario.core.scenario_state import ScenarioCoreState, ScenarioStateLike


def test_scenario_core_state_has_required_members():
    names = {member.name for member in ScenarioCoreState}
    assert names == {"UNINITIALIZED", "INITIALIZING", "EXECUTING", "COMPLETE", "FAILED"}


def test_scenario_core_state_satisfies_protocol():
    # Runtime-checkable protocol verifies presence of `name` and `value`.
    assert isinstance(ScenarioCoreState.UNINITIALIZED, ScenarioStateLike)


def test_per_scenario_subclass_enum_also_satisfies_protocol():
    class MyScenarioState(Enum):
        OPENING_PHASE = "opening_phase"
        ESCALATING = "escalating"

    assert isinstance(MyScenarioState.OPENING_PHASE, ScenarioStateLike)


def test_values_are_lower_snake_case_strings():
    for member in ScenarioCoreState:
        assert isinstance(member.value, str)
        assert member.value == member.name.lower()
