# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for ``pyrit.identifiers.step_identifier``."""

import pytest

from pyrit.identifiers import (
    STEP_EVAL_VERSION,
    ComponentIdentifier,
    build_step_identifier,
)


def _make_atomic_identifier(*, hash_suffix: str = "a") -> ComponentIdentifier:
    return ComponentIdentifier(
        class_name="AtomicAttack",
        class_module="pyrit.scenario.core.atomic_attack",
        params={"marker": hash_suffix},
    )


def test_build_step_identifier_returns_component_identifier():
    result = build_step_identifier(
        step_name="opening",
        outcome="violation",
        attack_execution_identifiers=[],
    )
    assert isinstance(result, ComponentIdentifier)


def test_class_name_is_scenario_step():
    result = build_step_identifier(
        step_name="opening",
        outcome="violation",
        attack_execution_identifiers=[],
    )
    assert result.class_name == "ScenarioStep"
    assert result.class_module == "pyrit.scenario.core.scenario_step"


def test_params_include_step_name_outcome_and_version():
    result = build_step_identifier(
        step_name="opening",
        outcome="violation",
        attack_execution_identifiers=[],
    )
    assert result.params["step_name"] == "opening"
    assert result.params["outcome"] == "violation"
    assert result.params["eval_version"] == STEP_EVAL_VERSION


def test_attack_executions_are_nested_under_children():
    atomic_a = _make_atomic_identifier(hash_suffix="a")
    atomic_b = _make_atomic_identifier(hash_suffix="b")

    result = build_step_identifier(
        step_name="opening",
        outcome="violation",
        attack_execution_identifiers=[atomic_a, atomic_b],
    )

    assert "attack_executions" in result.children
    nested = result.children["attack_executions"]
    assert isinstance(nested, list)
    assert isinstance(nested[0], ComponentIdentifier)
    assert isinstance(nested[1], ComponentIdentifier)
    assert nested[0].params["marker"] == "a"
    assert nested[1].params["marker"] == "b"


def test_empty_attack_executions_allowed():
    """Steps that record outcomes from external signals may produce no attacks."""
    result = build_step_identifier(
        step_name="external_signal",
        outcome="received",
        attack_execution_identifiers=[],
    )
    assert result.children["attack_executions"] == []


def test_rejects_empty_step_name():
    with pytest.raises(ValueError, match="step_name must be non-empty"):
        build_step_identifier(
            step_name="",
            outcome="violation",
            attack_execution_identifiers=[],
        )


def test_rejects_empty_outcome():
    with pytest.raises(ValueError, match="outcome must be non-empty"):
        build_step_identifier(
            step_name="opening",
            outcome="",
            attack_execution_identifiers=[],
        )


def test_hash_is_deterministic_for_same_inputs():
    atomic = _make_atomic_identifier()
    a = build_step_identifier(
        step_name="opening",
        outcome="violation",
        attack_execution_identifiers=[atomic],
    )
    b = build_step_identifier(
        step_name="opening",
        outcome="violation",
        attack_execution_identifiers=[atomic],
    )
    assert a.hash == b.hash


def test_hash_differs_when_outcome_differs():
    atomic = _make_atomic_identifier()
    violation = build_step_identifier(
        step_name="opening",
        outcome="violation",
        attack_execution_identifiers=[atomic],
    )
    refusal = build_step_identifier(
        step_name="opening",
        outcome="refusal",
        attack_execution_identifiers=[atomic],
    )
    assert violation.hash != refusal.hash


def test_hash_differs_when_step_name_differs():
    atomic = _make_atomic_identifier()
    opening = build_step_identifier(
        step_name="opening",
        outcome="violation",
        attack_execution_identifiers=[atomic],
    )
    escalating = build_step_identifier(
        step_name="escalating",
        outcome="violation",
        attack_execution_identifiers=[atomic],
    )
    assert opening.hash != escalating.hash


def test_step_eval_version_is_positive_int():
    assert isinstance(STEP_EVAL_VERSION, int)
    assert STEP_EVAL_VERSION >= 1


def test_hash_differs_when_attack_execution_child_config_differs():
    """Same step_name and outcome but different atomic child params must NOT dedup."""
    atomic_a = _make_atomic_identifier(hash_suffix="a")
    atomic_b = _make_atomic_identifier(hash_suffix="b")

    only_a = build_step_identifier(
        step_name="opening",
        outcome="violation",
        attack_execution_identifiers=[atomic_a],
    )
    only_b = build_step_identifier(
        step_name="opening",
        outcome="violation",
        attack_execution_identifiers=[atomic_b],
    )
    assert only_a.hash != only_b.hash


def test_attack_executions_value_is_list_not_nested_dict():
    """children["attack_executions"] must be a ``list[ComponentIdentifier]``, never a nested dict."""
    atomic = _make_atomic_identifier()
    result = build_step_identifier(
        step_name="opening",
        outcome="violation",
        attack_execution_identifiers=[atomic],
    )
    nested = result.children["attack_executions"]
    assert isinstance(nested, list)
    assert not isinstance(nested, dict)
    assert all(isinstance(c, ComponentIdentifier) for c in nested)


def test_attack_executions_preserves_input_order():
    """``attack_execution_identifiers`` is preserved in execution order (per docstring), so reversed inputs
    produce a different hash. This guards against an accidental sort that would erase execution-order
    semantics for branching/adaptive scenarios."""
    atomic_a = _make_atomic_identifier(hash_suffix="a")
    atomic_b = _make_atomic_identifier(hash_suffix="b")

    ab = build_step_identifier(
        step_name="opening",
        outcome="violation",
        attack_execution_identifiers=[atomic_a, atomic_b],
    )
    ba = build_step_identifier(
        step_name="opening",
        outcome="violation",
        attack_execution_identifiers=[atomic_b, atomic_a],
    )
    # Lists preserve caller-supplied order.
    assert [c.params["marker"] for c in ab.children["attack_executions"]] == ["a", "b"]
    assert [c.params["marker"] for c in ba.children["attack_executions"]] == ["b", "a"]
    # Hash reflects order — these are NOT considered the same step execution.
    assert ab.hash != ba.hash


def test_hash_changes_when_a_child_param_changes():
    """If an atomic child's params change, the step identifier's hash changes too."""
    baseline = build_step_identifier(
        step_name="opening",
        outcome="violation",
        attack_execution_identifiers=[_make_atomic_identifier(hash_suffix="v1")],
    )
    mutated = build_step_identifier(
        step_name="opening",
        outcome="violation",
        attack_execution_identifiers=[_make_atomic_identifier(hash_suffix="v2")],
    )
    assert baseline.hash != mutated.hash


def test_step_identifier_eval_version_in_params():
    """The schema/eval version is embedded in params so old rows preserve their original version."""
    result = build_step_identifier(
        step_name="opening",
        outcome="violation",
        attack_execution_identifiers=[],
    )
    assert "eval_version" in result.params
    assert result.params["eval_version"] == STEP_EVAL_VERSION
