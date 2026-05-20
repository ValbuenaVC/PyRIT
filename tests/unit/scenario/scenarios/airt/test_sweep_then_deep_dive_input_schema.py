# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Phase 8b — coverage for ``BroadSweepThenDeepDive.input_schema()`` + opacity flags."""

from __future__ import annotations

from pyrit.scenario.core.input_schema import RoleDescriptor, RoleTag
from pyrit.scenario.scenarios.airt.sweep_then_deep_dive import (
    BroadSweepThenDeepDive,
    CategoryAggregatingSweepStep,
    FilteredDeepDiveStep,
)


class TestBroadSweepThenDeepDiveInputSchema:
    """``BroadSweepThenDeepDive.input_schema()`` declares 3 OPAQUE roles + 1 SCALAR."""

    def test_returns_four_roles(self):
        schema = BroadSweepThenDeepDive.input_schema()
        assert len(schema) == 4
        assert all(isinstance(role, RoleDescriptor) for role in schema)

    def test_role_names_match_constructor_inputs(self):
        names = [role.name for role in BroadSweepThenDeepDive.input_schema()]
        assert names == [
            "sweep_atomic_attack",
            "deep_dive_atomic_attacks",
            "outcome_scorer",
            "weakness_label",
        ]

    def test_three_opaque_one_scalar(self):
        by_tag: dict[RoleTag, list[str]] = {tag: [] for tag in RoleTag}
        for role in BroadSweepThenDeepDive.input_schema():
            by_tag[role.tag].append(role.name)
        assert sorted(by_tag[RoleTag.OPAQUE]) == sorted(
            ["sweep_atomic_attack", "deep_dive_atomic_attacks", "outcome_scorer"]
        )
        assert by_tag[RoleTag.SCALAR] == ["weakness_label"]
        # No other tags present.
        for tag, names in by_tag.items():
            if tag not in {RoleTag.OPAQUE, RoleTag.SCALAR}:
                assert names == []

    def test_opaque_roles_are_required(self):
        """Each pre-built ``Identifiable`` instance is non-optional."""
        opaque_roles = [role for role in BroadSweepThenDeepDive.input_schema() if role.tag is RoleTag.OPAQUE]
        assert len(opaque_roles) == 3
        assert all(role.required is True for role in opaque_roles)

    def test_weakness_label_defaults_match_constructor(self):
        by_name = {role.name: role for role in BroadSweepThenDeepDive.input_schema()}
        weakness = by_name["weakness_label"]
        assert weakness.default == "safety_violation"
        assert weakness.required is False
        assert weakness.param_type is str

    def test_no_role_for_objective_scorer(self):
        """``objective_scorer`` is base-lifecycle plumbing, handled outside input_schema()."""
        names = {role.name for role in BroadSweepThenDeepDive.input_schema()}
        assert "objective_scorer" not in names

    def test_no_role_for_scenario_result_id(self):
        """``scenario_result_id`` is resume plumbing, never a wizard role."""
        names = {role.name for role in BroadSweepThenDeepDive.input_schema()}
        assert "scenario_result_id" not in names

    def test_descriptions_are_non_empty(self):
        for role in BroadSweepThenDeepDive.input_schema():
            assert role.description, f"role {role.name!r} has empty description"


class TestGraphArtifactOpaque:
    """8b-2 invariant: both branching-step classes flag themselves as artifact-opaque."""

    def test_category_aggregating_sweep_step_is_opaque(self):
        assert CategoryAggregatingSweepStep.GRAPH_ARTIFACT_OPAQUE is True

    def test_filtered_deep_dive_step_is_opaque(self):
        assert FilteredDeepDiveStep.GRAPH_ARTIFACT_OPAQUE is True

    def test_opacity_is_class_level_not_instance_level(self):
        """The flag must be readable from the class without instantiation."""
        assert hasattr(CategoryAggregatingSweepStep, "GRAPH_ARTIFACT_OPAQUE")
        assert hasattr(FilteredDeepDiveStep, "GRAPH_ARTIFACT_OPAQUE")
