# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Phase 8b — coverage for ``AdaptiveScenario.input_schema()`` + UUID determinism."""

from __future__ import annotations

import hashlib

import pytest

from pyrit.models import SeedAttackGroup, SeedObjective
from pyrit.scenario.core.input_schema import RoleDescriptor, RoleTag
from pyrit.scenario.scenarios.adaptive.adaptive_scenario import AdaptiveScenario


def _make_seed_group(*, value: str, harm_categories: list[str] | None = None) -> SeedAttackGroup:
    return SeedAttackGroup(seeds=[SeedObjective(value=value, harm_categories=harm_categories)])


class TestAdaptiveInputSchema:
    """``AdaptiveScenario.input_schema()`` declares the wizard-elicitable scalars."""

    def test_returns_four_roles(self):
        schema = AdaptiveScenario.input_schema()
        assert len(schema) == 4
        assert all(isinstance(role, RoleDescriptor) for role in schema)

    def test_role_names_match_constructor_scalars(self):
        names = [role.name for role in AdaptiveScenario.input_schema()]
        assert names == ["epsilon", "pool_threshold", "max_attempts_per_objective", "seed"]

    def test_all_roles_are_scalar(self):
        schema = AdaptiveScenario.input_schema()
        assert all(role.tag is RoleTag.SCALAR for role in schema)

    def test_defaults_mirror_constructor_signature(self):
        by_name = {role.name: role for role in AdaptiveScenario.input_schema()}
        assert by_name["epsilon"].default == 0.2
        assert by_name["pool_threshold"].default == 3
        assert by_name["max_attempts_per_objective"].default == 3
        assert by_name["seed"].default is None

    def test_param_types_match_constructor_annotations(self):
        by_name = {role.name: role for role in AdaptiveScenario.input_schema()}
        assert by_name["epsilon"].param_type is float
        assert by_name["pool_threshold"].param_type is int
        assert by_name["max_attempts_per_objective"].param_type is int
        assert by_name["seed"].param_type is int

    def test_all_roles_are_optional(self):
        """Every constructor scalar has a default; none are wizard-required."""
        schema = AdaptiveScenario.input_schema()
        assert all(role.required is False for role in schema)

    def test_no_role_for_objective_scorer(self):
        """``objective_scorer`` is base-lifecycle plumbing, handled outside input_schema()."""
        names = {role.name for role in AdaptiveScenario.input_schema()}
        assert "objective_scorer" not in names

    def test_no_role_for_context_extractor(self):
        """``context_extractor`` is a Callable; CLI cannot elicit it. Programmatic-only override."""
        names = {role.name for role in AdaptiveScenario.input_schema()}
        assert "context_extractor" not in names

    def test_no_role_for_scenario_strategies(self):
        """``scenario_strategies`` lives in ``supported_parameters()``, not input_schema()."""
        names = {role.name for role in AdaptiveScenario.input_schema()}
        assert "scenario_strategies" not in names


class TestObjectiveIdIsNonDeterministic:
    """8b-1 motivating invariant: ``SeedObjective.id`` auto-mints fresh UUIDs.

    This is why Phase 8b switched to content-hashing instead of trusting
    ``seed_group.objective.id`` for the atomic_attack_name suffix.
    """

    def test_seed_objective_id_default_is_fresh_uuid_per_construction(self):
        """``SeedObjective(value=...)`` mints a new UUID each call (see ``Seed.id`` default_factory).

        This is the root cause of the non-determinism that Phase 8b fixed:
        relying on ``objective.id`` for resume keys breaks across in-memory re-construction.
        """
        a = SeedObjective(value="cause harm")
        b = SeedObjective(value="cause harm")
        # Both have non-falsy ids (UUID4), but those ids are different.
        assert a.id
        assert b.id
        assert a.id != b.id

    def test_sha256_content_hash_is_stable_across_constructions(self):
        """The Phase 8b replacement (content hash) is deterministic by construction."""
        a = SeedObjective(value="cause harm")
        b = SeedObjective(value="cause harm")
        ha = hashlib.sha256(a.value.encode("utf-8")).hexdigest()[:12]
        hb = hashlib.sha256(b.value.encode("utf-8")).hexdigest()[:12]
        assert ha == hb
        assert len(ha) == 12

    def test_sha256_content_hash_differs_for_different_inputs(self):
        h1 = hashlib.sha256(b"obj-1").hexdigest()[:12]
        h2 = hashlib.sha256(b"obj-2").hexdigest()[:12]
        assert h1 != h2

    def test_uuid_module_no_longer_imported_in_adaptive_scenario(self):
        """``uuid`` is no longer imported by adaptive_scenario after Phase 8b.

        Phase 8b removed the ``uuid.uuid4()`` fallback and the
        ``seed_group.objective.id`` preference in favor of a deterministic
        SHA256 hash of the objective text. Re-adding ``uuid`` would re-introduce
        the non-determinism — this test guards against that regression by checking
        the module namespace directly.
        """
        import pyrit.scenario.scenarios.adaptive.adaptive_scenario as mod

        assert "uuid" not in mod.__dict__, (
            "uuid import was removed in Phase 8b in favor of deterministic SHA256 hashing; "
            "if you re-added it for a new use, also re-evaluate the deterministic-step-name invariant."
        )


@pytest.fixture
def two_identical_seed_groups() -> tuple[SeedAttackGroup, SeedAttackGroup]:
    """Two seed groups with identical objective text. Each gets its own ``objective.id`` UUID."""
    return (
        _make_seed_group(value="obj-deterministic", harm_categories=["violence"]),
        _make_seed_group(value="obj-deterministic", harm_categories=["violence"]),
    )


class TestContentHashRegressionSurface:
    """Pin the documented intent of the 8b-1 source change at a behavioral level.

    Note: a full ``_build_step_for_seed_group`` integration regression lives in
    ``test_text_adaptive.py::TestTextAdaptiveAtomicAttacks::test_atomic_names_are_deterministic_across_runs``.
    This class pins the lower-level invariants the integration test depends on.
    """

    def test_same_objective_value_yields_same_hash(self, two_identical_seed_groups):
        a, b = two_identical_seed_groups
        # Their auto-assigned ids differ (UUID4) — the hash is what makes resume stable.
        assert a.objective.id != b.objective.id
        hash_a = hashlib.sha256(a.objective.value.encode("utf-8")).hexdigest()[:12]
        hash_b = hashlib.sha256(b.objective.value.encode("utf-8")).hexdigest()[:12]
        assert hash_a == hash_b
