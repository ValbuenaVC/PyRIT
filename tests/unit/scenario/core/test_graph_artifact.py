# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Phase 8g — coverage for ``graph_artifact`` build / serialize / load primitives."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from pyrit.scenario.core.graph_artifact import (
    GraphArtifact,
    GraphArtifactDriftError,
    GraphArtifactError,
    GraphArtifactSecurityError,
    OpaqueInputUnresolvedError,
    _canonical_json,
    _class_fqn,
    _deserialize_dataset_config,
    _encode_init_inputs,
    _resolve_scenario_fqn,
    _serialize_dataset_config,
    _topology_hash,
    build_graph_artifact,
    build_topology_summary,
    graph_artifact_from_yaml,
    graph_artifact_to_yaml,
    materialize_opaque_inputs,
)
from pyrit.scenario.core.input_schema import RoleDescriptor, RoleTag

# --- fake scenario surface ------------------------------------------------------
#
# build_graph_artifact reads a small slice of the Scenario lifecycle. We model
# that slice with a real class so ``type(scenario)`` round-trips through
# ``_class_fqn`` and class methods (``input_schema``, ``get_strategy_class``)
# work without monkeying with ``__class__``.


class _FakeStrategyEnum:
    """Stand-in for a ScenarioStrategy enum class — only needs an FQN."""


class _FakeScenario:
    """Minimal Scenario-shaped object for graph_artifact unit tests."""

    @classmethod
    def input_schema(cls) -> list[RoleDescriptor]:
        return []

    @classmethod
    def get_strategy_class(cls) -> type:
        return _FakeStrategyEnum


def _fake_atomic(*, name: str, hash_value: str = "abc123") -> MagicMock:
    """A mock that satisfies ``atomic.get_identifier().to_dict()``."""
    atomic = MagicMock()
    identifier = MagicMock()
    identifier.to_dict.return_value = {"name": name, "hash": hash_value}
    atomic.get_identifier.return_value = identifier
    return atomic


def _fake_initialized_scenario(
    *,
    version: int = 1,
    atomic_names: tuple[str, ...] = ("step_a", "step_b"),
    states: tuple[str, ...] = ("STATE_0", "STATE_1"),
    initial_state: str = "STATE_0",
    terminal_states: tuple[str, ...] = ("STATE_DONE",),
    strategies: tuple[str, ...] = ("EASY",),
    dataset_names: list[str] | None = None,
    max_dataset_size: int | None = None,
    include_baseline: bool = False,
    params: dict[str, Any] | None = None,
    memory_labels: dict[str, str] | None = None,
) -> _FakeScenario:
    """
    A lightly-mocked stand-in for a fully-initialized ``Scenario``.

    The graph artifact build path only reads a small surface area — we mirror
    that surface without spinning up the real lifecycle.
    """
    scenario = _FakeScenario()

    identifier = MagicMock()
    identifier.version = version
    scenario._identifier = identifier  # type: ignore[attr-defined]

    strategy_objs = []
    for name in strategies:
        s = MagicMock()
        s.name = name
        strategy_objs.append(s)
    scenario._scenario_strategies = strategy_objs  # type: ignore[attr-defined]

    cfg = MagicMock()
    cfg._dataset_names = dataset_names
    cfg.max_dataset_size = max_dataset_size
    cfg._seed_groups = None
    scenario._dataset_config = cfg  # type: ignore[attr-defined]

    scenario._include_baseline = include_baseline  # type: ignore[attr-defined]
    scenario.params = params or {}  # type: ignore[attr-defined]
    scenario._memory_labels = memory_labels or {}  # type: ignore[attr-defined]

    scenario._atomic_attacks = [_fake_atomic(name=n) for n in atomic_names]  # type: ignore[attr-defined]

    policy = MagicMock()
    policy.actions = {s: (lambda _g: None) for s in states}
    policy.initial_state = initial_state
    policy.terminal_states = frozenset(terminal_states)
    graph = MagicMock()
    graph.policy = policy
    scenario.execution_graph = graph  # type: ignore[attr-defined]

    return scenario


# --- _canonical_json -------------------------------------------------------------


class TestCanonicalJson:
    def test_sorts_keys(self):
        assert _canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'

    def test_nested_sort(self):
        assert _canonical_json({"b": {"y": 1, "x": 2}}) == '{"b":{"x":2,"y":1}}'

    def test_handles_unserializable_via_str(self):
        class _X:
            def __repr__(self) -> str:
                return "<X>"

        assert _canonical_json({"obj": _X()}) == '{"obj":"<X>"}'


# --- _class_fqn ------------------------------------------------------------------


class TestClassFqn:
    def test_returns_module_dot_qualname(self):
        # Use the module-level _FakeScenario so we get a clean, predictable qualname.
        fqn = _class_fqn(_FakeScenario)
        assert fqn.endswith("._FakeScenario")
        assert "." in fqn  # must be a dotted FQN, not a bare class name


# --- _resolve_scenario_fqn -------------------------------------------------------


class TestResolveScenarioFqn:
    def test_resolves_registered_scenario(self):
        cls = _resolve_scenario_fqn("pyrit.scenario.scenarios.garak.encoding.Encoding")
        assert cls.__name__ == "Encoding"

    def test_rejects_non_dotted_fqn(self):
        with pytest.raises(GraphArtifactSecurityError, match="not a dotted path"):
            _resolve_scenario_fqn("AdaptiveScenario")

    def test_rejects_unimportable_module(self):
        with pytest.raises(GraphArtifactSecurityError):
            _resolve_scenario_fqn("nonexistent.module.path.SomeClass")

    def test_rejects_missing_attribute(self):
        with pytest.raises(GraphArtifactSecurityError):
            _resolve_scenario_fqn("pyrit.scenario.core.scenario.NotAClass")

    def test_rejects_non_scenario_subclass(self):
        with pytest.raises(GraphArtifactSecurityError, match="not a Scenario subclass"):
            _resolve_scenario_fqn("pyrit.scenario.core.input_schema.RoleDescriptor")

    def test_rejects_unregistered_scenario_subclass(self):
        """A real Scenario subclass that's not registry-discoverable must be rejected."""

        # Define a private Scenario subclass at module load time — the registry
        # only discovers scenarios in pyrit.scenario.scenarios.*, so this should
        # be rejected even though it IS a Scenario subclass.
        # We can't easily inject one without polluting the registry, so we use
        # the abstract `Scenario` itself which is not registered.
        with pytest.raises(GraphArtifactSecurityError, match="not in the registry whitelist"):
            _resolve_scenario_fqn("pyrit.scenario.core.scenario.Scenario")


# --- DatasetConfiguration round-trip --------------------------------------------


class TestDatasetConfigSerialize:
    def test_serializes_dataset_names(self):
        cfg = MagicMock()
        cfg._dataset_names = ["xstest"]
        cfg.max_dataset_size = 10
        cfg._seed_groups = None
        out = _serialize_dataset_config(cfg)
        assert out == {"dataset_names": ["xstest"], "max_dataset_size": 10, "explicit_seed_groups_count": 0}

    def test_serializes_none_dataset_names(self):
        cfg = MagicMock()
        cfg._dataset_names = None
        cfg.max_dataset_size = None
        cfg._seed_groups = None
        out = _serialize_dataset_config(cfg)
        assert out["dataset_names"] is None
        assert out["max_dataset_size"] is None

    def test_records_explicit_seed_group_count(self):
        cfg = MagicMock()
        cfg._dataset_names = None
        cfg.max_dataset_size = None
        cfg._seed_groups = [MagicMock(), MagicMock(), MagicMock()]
        out = _serialize_dataset_config(cfg)
        assert out["explicit_seed_groups_count"] == 3


class TestDatasetConfigDeserialize:
    def test_round_trips_dataset_names(self):
        cfg = _deserialize_dataset_config(
            {"dataset_names": ["xstest"], "max_dataset_size": 10, "explicit_seed_groups_count": 0}
        )
        assert cfg is not None
        assert cfg._dataset_names == ["xstest"]
        assert cfg.max_dataset_size == 10

    def test_empty_payload_returns_none(self):
        assert _deserialize_dataset_config({}) is None

    def test_explicit_seed_groups_raises(self):
        with pytest.raises(GraphArtifactError, match="explicit seed_groups"):
            _deserialize_dataset_config({"explicit_seed_groups_count": 2})


# --- _topology_hash --------------------------------------------------------------


class TestTopologyHash:
    def test_deterministic_across_calls(self):
        summary = {"states": ["A", "B"], "initial_state": "A"}
        assert _topology_hash(summary) == _topology_hash(summary)

    def test_changes_when_summary_changes(self):
        a = _topology_hash({"states": ["A"]})
        b = _topology_hash({"states": ["A", "B"]})
        assert a != b


# --- build_topology_summary ------------------------------------------------------


class TestBuildTopologySummary:
    def test_collects_states_atoms_and_terminals(self):
        scenario = _fake_initialized_scenario(atomic_names=("step_x",), states=("S0",), terminal_states=("DONE",))
        summary = build_topology_summary(scenario)
        assert summary["states"] == ["S0"]
        assert summary["initial_state"] == "STATE_0"
        assert summary["terminal_states"] == ["DONE"]
        assert len(summary["atomic_attacks"]) == 1
        assert summary["atomic_attacks"][0] == {"name": "step_x", "hash": "abc123"}

    def test_sorted_state_lists_for_determinism(self):
        scenario = _fake_initialized_scenario(states=("S_B", "S_A", "S_C"), terminal_states=("T_B", "T_A"))
        summary = build_topology_summary(scenario)
        assert summary["states"] == ["S_A", "S_B", "S_C"]
        assert summary["terminal_states"] == ["T_A", "T_B"]

    def test_includes_scenario_class_and_version(self):
        scenario = _fake_initialized_scenario(version=7)
        summary = build_topology_summary(scenario)
        assert summary["scenario_version"] == 7
        assert "scenario_class_fqn" in summary


# --- _encode_init_inputs ---------------------------------------------------------


class TestEncodeInitInputs:
    def test_scalar_passes_through(self):
        schema = [RoleDescriptor(name="x", description="d", tag=RoleTag.SCALAR, param_type=str)]
        out = _encode_init_inputs(schema=schema, init_inputs={"x": "hello"})
        assert out == {"x": "hello"}

    def test_opaque_value_snapshots_via_identifier(self):
        instance = MagicMock()
        instance.get_identifier.return_value.to_dict.return_value = {"cls": "Atomic", "hash": "h1"}
        schema = [RoleDescriptor(name="atom", description="d", tag=RoleTag.OPAQUE)]
        out = _encode_init_inputs(schema=schema, init_inputs={"atom": instance})
        assert out == {"atom": {"cls": "Atomic", "hash": "h1"}}

    def test_opaque_value_without_identifier_passes_through(self):
        instance = object()  # no get_identifier attribute
        schema = [RoleDescriptor(name="atom", description="d", tag=RoleTag.OPAQUE)]
        out = _encode_init_inputs(schema=schema, init_inputs={"atom": instance})
        assert out["atom"] is instance

    def test_opaque_none_passes_through(self):
        schema = [RoleDescriptor(name="atom", description="d", tag=RoleTag.OPAQUE, required=False, default="x")]
        out = _encode_init_inputs(schema=schema, init_inputs={"atom": None})
        assert out["atom"] is None

    def test_unknown_input_passes_through(self):
        out = _encode_init_inputs(schema=[], init_inputs={"unknown_kwarg": 5})
        assert out == {"unknown_kwarg": 5}


# --- build_graph_artifact --------------------------------------------------------


class TestBuildGraphArtifact:
    def test_populates_all_fields(self):
        scenario = _fake_initialized_scenario(
            version=3,
            strategies=("EASY", "HARD"),
            dataset_names=["xstest"],
            max_dataset_size=42,
            include_baseline=True,
            params={"alpha": 0.5},
            memory_labels={"run": "abc"},
        )
        artifact = build_graph_artifact(scenario)
        assert artifact.scenario_class_fqn.endswith("._FakeScenario")
        assert artifact.scenario_version == 3
        assert artifact.scenario_strategies == ["EASY", "HARD"]
        assert artifact.dataset_config["dataset_names"] == ["xstest"]
        assert artifact.dataset_config["max_dataset_size"] == 42
        assert artifact.include_baseline is True
        assert artifact.params == {"alpha": 0.5}
        assert artifact.memory_labels == {"run": "abc"}
        assert artifact.topology_hash != ""
        assert artifact.state_enum_fqn is not None
        assert artifact.state_enum_fqn.endswith("._FakeStrategyEnum")

    def test_init_async_inputs_default_to_params(self):
        scenario = _fake_initialized_scenario(params={"alpha": 1.0})
        artifact = build_graph_artifact(scenario)
        assert artifact.init_async_inputs == {"alpha": 1.0}

    def test_explicit_init_async_inputs_override_params(self):
        scenario = _fake_initialized_scenario(params={"alpha": 1.0})
        artifact = build_graph_artifact(scenario, init_async_inputs={"max_concurrency": 4})
        assert artifact.init_async_inputs == {"max_concurrency": 4}

    def test_topology_hash_stable_across_builds(self):
        s1 = _fake_initialized_scenario()
        s2 = _fake_initialized_scenario()
        assert build_graph_artifact(s1).topology_hash == build_graph_artifact(s2).topology_hash


# --- YAML round-trip -------------------------------------------------------------


class TestYamlRoundTrip:
    def test_to_and_from_yaml(self, tmp_path):
        scenario = _fake_initialized_scenario(version=5)
        artifact = build_graph_artifact(scenario)
        path = tmp_path / "artifact.yaml"
        graph_artifact_to_yaml(artifact, path)
        loaded = graph_artifact_from_yaml(path)
        # field-for-field equivalence except set/frozenset reshape (none in artifact).
        assert loaded == artifact

    def test_byte_identical_for_equal_artifacts(self, tmp_path):
        s1 = _fake_initialized_scenario(params={"a": 1, "b": 2})
        s2 = _fake_initialized_scenario(params={"b": 2, "a": 1})
        a1 = build_graph_artifact(s1)
        a2 = build_graph_artifact(s2)
        p1 = tmp_path / "a1.yaml"
        p2 = tmp_path / "a2.yaml"
        graph_artifact_to_yaml(a1, p1)
        graph_artifact_to_yaml(a2, p2)
        assert p1.read_bytes() == p2.read_bytes()

    def test_yaml_payload_is_sort_key_canonical(self, tmp_path):
        scenario = _fake_initialized_scenario()
        artifact = build_graph_artifact(scenario)
        path = tmp_path / "a.yaml"
        graph_artifact_to_yaml(artifact, path)
        text = path.read_text(encoding="utf-8")
        # First top-level key should be alphabetically smallest, i.e. "artifact_version".
        first_key = text.splitlines()[0].split(":", 1)[0]
        assert first_key == "artifact_version"

    def test_from_yaml_rejects_wrong_artifact_version(self, tmp_path):
        path = tmp_path / "stale.yaml"
        # Build a valid artifact then rewrite its artifact_version to a stale value.
        scenario = _fake_initialized_scenario()
        artifact = build_graph_artifact(scenario)
        graph_artifact_to_yaml(artifact, path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payload["artifact_version"] = 99
        path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
        with pytest.raises(GraphArtifactError, match="artifact_version"):
            graph_artifact_from_yaml(path)

    def test_from_yaml_rejects_non_mapping(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- not a mapping", encoding="utf-8")
        with pytest.raises(GraphArtifactError, match="not a mapping"):
            graph_artifact_from_yaml(path)


# --- materialize_opaque_inputs ---------------------------------------------------


class _DummyScenarioCls:
    """Provides ``input_schema`` for ``materialize_opaque_inputs`` tests."""

    _schema: list[RoleDescriptor] = []

    @classmethod
    def input_schema(cls) -> list[RoleDescriptor]:
        return cls._schema


class TestMaterializeOpaqueInputs:
    def test_scalar_passes_through(self):
        _DummyScenarioCls._schema = [
            RoleDescriptor(name="alpha", description="d", tag=RoleTag.SCALAR, param_type=float)
        ]
        out = materialize_opaque_inputs(_DummyScenarioCls, {"alpha": 0.5})  # type: ignore[arg-type]
        assert out == {"alpha": 0.5}

    def test_opaque_live_instance_passes_through(self):
        instance = MagicMock()
        _DummyScenarioCls._schema = [RoleDescriptor(name="atom", description="d", tag=RoleTag.OPAQUE)]
        out = materialize_opaque_inputs(_DummyScenarioCls, {"atom": instance})  # type: ignore[arg-type]
        assert out["atom"] is instance

    def test_opaque_dict_with_materializer_rebuilds(self):
        rebuilt = object()
        _DummyScenarioCls._schema = [RoleDescriptor(name="atom", description="d", tag=RoleTag.OPAQUE)]
        out = materialize_opaque_inputs(
            _DummyScenarioCls,  # type: ignore[arg-type]
            {"atom": {"hash": "h"}},
            opaque_materializers={"atom": lambda payload: rebuilt},
        )
        assert out["atom"] is rebuilt

    def test_opaque_dict_without_materializer_raises(self):
        _DummyScenarioCls._schema = [RoleDescriptor(name="atom", description="d", tag=RoleTag.OPAQUE)]
        with pytest.raises(OpaqueInputUnresolvedError) as exc:
            materialize_opaque_inputs(_DummyScenarioCls, {"atom": {"hash": "h"}})  # type: ignore[arg-type]
        assert exc.value.role_name == "atom"
        assert "opaque_materializers" in str(exc.value)


# --- GraphArtifact dataclass shape ----------------------------------------------


class TestGraphArtifactDataclass:
    def test_frozen(self):
        artifact = GraphArtifact(scenario_class_fqn="x.Y", scenario_version=1, pyrit_version="0.0.0")
        with pytest.raises(Exception):
            artifact.scenario_version = 2  # type: ignore[misc]

    def test_default_field_values(self):
        artifact = GraphArtifact(scenario_class_fqn="x.Y", scenario_version=1, pyrit_version="0.0.0")
        assert artifact.init_inputs == {}
        assert artifact.scenario_strategies == []
        assert artifact.include_baseline is False
        assert artifact.topology_hash == ""
        assert artifact.state_enum_fqn is None
        assert artifact.artifact_version == 1


# --- Error-type shape ------------------------------------------------------------


class TestErrorTypes:
    def test_security_error_is_artifact_error(self):
        assert issubclass(GraphArtifactSecurityError, GraphArtifactError)

    def test_drift_error_is_artifact_error(self):
        assert issubclass(GraphArtifactDriftError, GraphArtifactError)

    def test_opaque_unresolved_is_artifact_error(self):
        assert issubclass(OpaqueInputUnresolvedError, GraphArtifactError)

    def test_opaque_unresolved_carries_role_and_payload(self):
        payload = {"k": 1}
        exc = OpaqueInputUnresolvedError("alpha", payload)
        assert exc.role_name == "alpha"
        assert exc.payload is payload
