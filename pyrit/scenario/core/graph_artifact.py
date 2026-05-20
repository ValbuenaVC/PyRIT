# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Graph artifact — reproducible config capture for Python-authored scenarios.

A :class:`GraphArtifact` is a fully serializable snapshot of an initialized
scenario's *configuration* (the inputs you'd give the wizard) plus a
*topology snapshot* (what the underlying graph looked like at build time).
Reloading replays the configuration through
:func:`pyrit.scenario.core.builder.build_scenario_from_inputs`, then asserts
the rebuilt graph matches the snapshot.

Scope (deliberately narrow): this enables ``pyrit_scan --from-artifact path.yaml``
to reproduce a wizard-built scenario. It does NOT enable authoring new
scenario topologies from YAML; transitions live in Python closures inside
``_build_execution_graph`` and are not serialized. Drift detection is
*structural* (state set, step identifiers) not behavioral (predicate bodies).

Security: the load path resolves the captured scenario class FQN through
:class:`pyrit.registry.class_registries.scenario_registry.ScenarioRegistry`'s
self-discovered whitelist. Unregistered FQNs are rejected. Registered
scenarios can still run arbitrary code in ``__init__`` — this is acceptable
for PyRIT's trusted-developer threat model and documented here loudly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

import pyrit
from pyrit.scenario.core.builder import build_scenario_from_inputs
from pyrit.scenario.core.input_schema import RoleDescriptor, RoleTag

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pyrit.prompt_target import PromptTarget
    from pyrit.scenario.core.dataset_configuration import DatasetConfiguration
    from pyrit.scenario.core.scenario import Scenario


_ARTIFACT_VERSION = 1


class GraphArtifactError(Exception):
    """Base class for graph-artifact failures."""


class GraphArtifactSecurityError(GraphArtifactError):
    """Raised when an artifact's ``scenario_class_fqn`` is not registry-whitelisted."""


class GraphArtifactDriftError(GraphArtifactError):
    """Raised when a loaded scenario's rebuilt topology disagrees with the artifact snapshot."""


class OpaqueInputUnresolvedError(GraphArtifactError):
    """Raised when an opaque role's stored payload cannot be rematerialized at load time."""

    def __init__(self, role_name: str, payload: Any) -> None:
        """Format a help message naming the role and pointing at ``opaque_materializers``."""
        super().__init__(
            f"Opaque role {role_name!r} has a stored identifier payload but no materializer "
            "was supplied. Pass ``opaque_materializers={role_name: callable}`` to "
            "``load_scenario_from_artifact`` to rebuild the instance from its identifier."
        )
        self.role_name = role_name
        self.payload = payload


# --- helpers ---------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    """Return a deterministic JSON encoding suitable for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _class_fqn(cls: type) -> str:
    """Return the canonical ``module.qualname`` FQN for ``cls``."""
    return f"{cls.__module__}.{cls.__qualname__}"


def _resolve_scenario_fqn(fqn: str) -> type[Scenario]:
    """
    Resolve a captured FQN to a live :class:`Scenario` class.

    Security: the resolved class MUST appear in the discovered
    :class:`ScenarioRegistry` to be accepted. Unregistered FQNs are rejected
    even if the FQN points at a real Python class.

    Returns:
        type[Scenario]: The resolved scenario class.

    Raises:
        GraphArtifactSecurityError: If the FQN is not dotted, cannot be
            imported, does not resolve to a class, is not a ``Scenario``
            subclass, or is not in the registry whitelist.
    """
    from pyrit.registry.class_registries.scenario_registry import ScenarioRegistry
    from pyrit.scenario.core.scenario import Scenario

    module_path, _, class_name = fqn.rpartition(".")
    if not module_path:
        raise GraphArtifactSecurityError(f"Scenario FQN {fqn!r} is not a dotted path.")

    try:
        module = import_module(module_path)
        cls = getattr(module, class_name, None)
    except ImportError as exc:
        raise GraphArtifactSecurityError(f"Failed to import module for FQN {fqn!r}: {exc}") from exc

    if cls is None:
        raise GraphArtifactSecurityError(f"FQN {fqn!r} did not resolve to a class.")

    if not isinstance(cls, type) or not issubclass(cls, Scenario):
        raise GraphArtifactSecurityError(f"FQN {fqn!r} resolved to {cls!r}, which is not a Scenario subclass.")

    registry = ScenarioRegistry()
    registry._ensure_discovered()
    registered_classes = {entry.registered_class for entry in registry._class_entries.values()}
    if cls not in registered_classes:
        raise GraphArtifactSecurityError(
            f"Scenario class {fqn!r} is not in the registry whitelist. "
            "Only registered scenarios can be loaded from artifacts."
        )

    return cls


def _serialize_dataset_config(cfg: DatasetConfiguration) -> dict[str, Any]:
    """
    Serialize a :class:`DatasetConfiguration` to a JSON-compatible dict.

    Only ``dataset_names`` and ``max_dataset_size`` are captured. Explicit
    ``seed_groups`` are NOT serialized in 8g MVP — artifacts built from
    explicit-seed-group configs will fail to fully round-trip if the caller
    doesn't re-supply equivalent groups via load-time overrides.

    Returns:
        dict[str, Any]: A JSON-compatible mapping carrying ``dataset_names``,
            ``max_dataset_size``, and a marker count for explicit seed groups.
    """
    if cfg is None:
        return {}
    return {
        "dataset_names": list(cfg._dataset_names) if cfg._dataset_names is not None else None,
        "max_dataset_size": cfg.max_dataset_size,
        "explicit_seed_groups_count": len(cfg._seed_groups) if cfg._seed_groups else 0,
    }


def _deserialize_dataset_config(payload: Mapping[str, Any]) -> DatasetConfiguration | None:
    """
    Reconstruct a :class:`DatasetConfiguration` from a serialized payload.

    Returns:
        DatasetConfiguration | None: A reconstructed config, or ``None`` when
            the payload is empty (i.e. the original scenario had no dataset
            configuration to capture).

    Raises:
        GraphArtifactError: If the payload claims explicit seed groups, which
            8g MVP does not support round-tripping.
    """
    from pyrit.scenario.core.dataset_configuration import DatasetConfiguration

    if not payload:
        return None
    if payload.get("explicit_seed_groups_count", 0) > 0:
        raise GraphArtifactError(
            "Artifact captured a DatasetConfiguration with explicit seed_groups, which 8g MVP "
            "does not round-trip. Supply seed_groups via a build-time override or rebuild the "
            "artifact from a dataset_names-based configuration."
        )
    return DatasetConfiguration(
        dataset_names=payload.get("dataset_names"),
        max_dataset_size=payload.get("max_dataset_size"),
    )


# --- dataclass -------------------------------------------------------------------


@dataclass(frozen=True)
class GraphArtifact:
    """
    Reproducible snapshot of an initialized scenario's configuration + topology.

    Attributes:
        scenario_class_fqn: ``module.ClassName`` for whitelist-resolved load.
        scenario_version: Mirrors the scenario's instance-time ``version=`` arg
            (read from ``scenario._identifier.scenario_version``).
        pyrit_version: Stamped from ``pyrit.__version__`` at build time.
        artifact_version: Bumped only on backward-incompatible schema changes.
        init_inputs: Validated against ``input_schema()``. Opaque values are
            stored as ``value.get_identifier().to_dict()`` payloads.
        init_async_inputs: Scalar arguments forwarded to ``initialize_async``.
            Typically the contents of ``self.params``.
        scenario_strategies: Enum-member names of the strategies selected at
            initialize time (e.g. ``["EASY", "HARD"]``).
        dataset_config: Serialized :class:`DatasetConfiguration` (subset).
        include_baseline: Resolved boolean from the scenario's effective
            :class:`BaselineAttackPolicy`.
        params: Snapshot of ``self.params`` after ``set_params_from_args``.
        memory_labels: Run-time memory labels.
        topology_hash: ``sha256(canonical_json(topology_summary))`` for drift
            comparison at load time.
        topology_summary: Human-readable structural snapshot (states, terminals,
            atomic-attack identifiers).
        state_enum_fqn: ``module.Enum`` for branching scenarios that use an
            Enum state type; ``None`` for legacy ``int``-keyed linear scenarios.
    """

    scenario_class_fqn: str
    scenario_version: int
    pyrit_version: str
    artifact_version: int = _ARTIFACT_VERSION

    init_inputs: dict[str, Any] = field(default_factory=dict)
    init_async_inputs: dict[str, Any] = field(default_factory=dict)

    scenario_strategies: list[str] = field(default_factory=list)
    dataset_config: dict[str, Any] = field(default_factory=dict)
    include_baseline: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    memory_labels: dict[str, str] = field(default_factory=dict)

    topology_hash: str = ""
    topology_summary: dict[str, Any] = field(default_factory=dict)
    state_enum_fqn: str | None = None


# --- build path ------------------------------------------------------------------


def build_topology_summary(scenario: Scenario) -> dict[str, Any]:
    """
    Produce a deterministic structural snapshot of ``scenario.execution_graph``.

    Captures states, initial / terminal state names, and the atomic-attack
    identifier list. Per-state step bindings are NOT recorded (they live in
    closure bodies inside the policy actions and cannot be introspected).
    Drift checks therefore catch state-set changes, atomic-attack changes, and
    policy-initial / terminal changes — not behavioral changes inside
    transition predicates.

    Args:
        scenario: A scenario that has completed ``initialize_async`` (so
            ``execution_graph`` and ``_atomic_attacks`` are populated).

    Returns:
        dict[str, Any]: A JSON-compatible mapping suitable for hashing.

    Raises:
        ValueError: If the scenario has not been initialized (no execution graph).
    """
    if scenario.execution_graph is None:
        # Fall back to building the graph from current atomic attacks so callers
        # can snapshot a scenario whose run_async hasn't been invoked yet.
        scenario._execution_graph = scenario._build_execution_graph(steps=scenario._atomic_attacks)

    graph = scenario.execution_graph
    assert graph is not None  # narrowed above
    policy = graph.policy

    return {
        "scenario_class_fqn": _class_fqn(type(scenario)),
        "scenario_version": scenario._identifier.version,
        "states": sorted([str(state) for state in policy.actions]),
        "initial_state": str(policy.initial_state),
        "terminal_states": sorted([str(state) for state in policy.terminal_states]),
        "atomic_attacks": [atomic.get_identifier().to_dict() for atomic in scenario._atomic_attacks],
    }


def _topology_hash(summary: Mapping[str, Any]) -> str:
    """
    Compute the canonical sha256 hash for a topology summary.

    Returns:
        str: The hex digest of ``sha256(_canonical_json(summary))``.
    """
    return hashlib.sha256(_canonical_json(summary).encode("utf-8")).hexdigest()


def _encode_init_inputs(
    *,
    schema: list[RoleDescriptor],
    init_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Encode init_inputs for serialization, snapshotting OPAQUE roles as identifier dicts.

    Returns:
        dict[str, Any]: A serialization-safe mapping with opaque live instances
            replaced by their ``ComponentIdentifier.to_dict()`` payloads.
    """
    encoded: dict[str, Any] = {}
    schema_by_name = {role.name: role for role in schema}
    for name, value in init_inputs.items():
        role = schema_by_name.get(name)
        if role is not None and role.tag is RoleTag.OPAQUE and value is not None:
            if hasattr(value, "get_identifier"):
                encoded[name] = value.get_identifier().to_dict()
            else:
                # Unknown opaque shape — defer to caller serialization at their own risk.
                encoded[name] = value
        else:
            encoded[name] = value
    return encoded


def build_graph_artifact(
    scenario: Scenario,
    *,
    init_inputs: Mapping[str, Any] | None = None,
    init_async_inputs: Mapping[str, Any] | None = None,
) -> GraphArtifact:
    """
    Snapshot an initialized scenario as a :class:`GraphArtifact`.

    Args:
        scenario: Must have completed ``initialize_async``.
        init_inputs: The rich-object ``__init__`` arguments the scenario was
            built with. Required because ``Scenario`` does not store its
            constructor args directly — they're embedded in opaque attributes
            (``objective_scorer``, ``strategy_class``) that aren't easy to
            recover after construction.
        init_async_inputs: The scalar ``initialize_async`` arguments. Optional;
            defaults to ``self.params``.

    Returns:
        GraphArtifact: A frozen snapshot ready for YAML serialization.
    """
    init_inputs = dict(init_inputs or {})
    init_async_inputs = dict(init_async_inputs or scenario.params or {})

    schema = list(type(scenario).input_schema())
    encoded_init_inputs = _encode_init_inputs(schema=schema, init_inputs=init_inputs)

    strategy_names = [
        strategy.name if hasattr(strategy, "name") else str(strategy) for strategy in scenario._scenario_strategies
    ]
    strategy_cls = type(scenario).get_strategy_class()
    state_enum_fqn = _class_fqn(strategy_cls) if strategy_cls is not None else None

    topology = build_topology_summary(scenario)

    return GraphArtifact(
        scenario_class_fqn=_class_fqn(type(scenario)),
        scenario_version=scenario._identifier.version,
        pyrit_version=pyrit.__version__,
        init_inputs=encoded_init_inputs,
        init_async_inputs=init_async_inputs,
        scenario_strategies=strategy_names,
        dataset_config=_serialize_dataset_config(scenario._dataset_config),
        include_baseline=scenario._include_baseline,
        params=dict(scenario.params),
        memory_labels=dict(scenario._memory_labels),
        topology_hash=_topology_hash(topology),
        topology_summary=topology,
        state_enum_fqn=state_enum_fqn,
    )


# --- YAML I/O --------------------------------------------------------------------


def graph_artifact_to_yaml(artifact: GraphArtifact, path: str | Path) -> None:
    """
    Write ``artifact`` to ``path`` in deterministic YAML form.

    Two artifacts that compare equal at the dataclass level write byte-identical
    YAML thanks to ``sort_keys=True`` and ``default_flow_style=False``. This is
    the contract integration tests rely on for the byte-identical-output gate.
    """
    payload = asdict(artifact)
    Path(path).write_text(
        yaml.safe_dump(payload, sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )


def graph_artifact_from_yaml(path: str | Path) -> GraphArtifact:
    """
    Read a :class:`GraphArtifact` back from ``path``.

    Performs an ``artifact_version`` check so older artifacts surface a clear
    error instead of silently mismatching the dataclass schema.

    Returns:
        GraphArtifact: The deserialized artifact.

    Raises:
        GraphArtifactError: If the YAML payload is not a mapping, or if its
            ``artifact_version`` does not match the version this PyRIT
            understands.
    """
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GraphArtifactError(f"Artifact at {path!r} is not a mapping; got {type(payload).__name__}.")

    artifact_version = payload.get("artifact_version", 0)
    if artifact_version != _ARTIFACT_VERSION:
        raise GraphArtifactError(
            f"Artifact at {path!r} has artifact_version={artifact_version}, "
            f"but this PyRIT understands artifact_version={_ARTIFACT_VERSION}. "
            "Rebuild the artifact with the current version."
        )

    return GraphArtifact(**payload)


# --- load path -------------------------------------------------------------------


def materialize_opaque_inputs(
    cls: type[Scenario],
    init_inputs: Mapping[str, Any],
    *,
    opaque_materializers: Mapping[str, Callable[[dict[str, Any]], Any]] | None = None,
) -> dict[str, Any]:
    """
    Rebuild live opaque instances from their stored identifier payloads.

    For each OPAQUE role in ``cls.input_schema()``:

    * If the stored value is already a live instance (not a dict), pass through.
    * If the stored value is a dict and a callable is registered in
      ``opaque_materializers[role.name]``, invoke it with the dict.
    * Otherwise raise :class:`OpaqueInputUnresolvedError`.

    SCALAR / CHOICE roles pass through unchanged.

    Args:
        cls: The scenario class.
        init_inputs: Stored init inputs from the artifact.
        opaque_materializers: Optional mapping from role name to a callable
            that consumes the stored identifier dict and returns a live instance.

    Returns:
        dict[str, Any]: Materialized init inputs ready for
            :func:`build_scenario_from_inputs`.

    Raises:
        OpaqueInputUnresolvedError: When an opaque role has a dict payload but
            no materializer is provided.
    """
    materializers = dict(opaque_materializers or {})
    schema = {role.name: role for role in cls.input_schema()}
    out: dict[str, Any] = {}
    for name, value in init_inputs.items():
        role = schema.get(name)
        if role is not None and role.tag is RoleTag.OPAQUE and isinstance(value, dict):
            if name not in materializers:
                raise OpaqueInputUnresolvedError(name, value)
            out[name] = materializers[name](value)
        else:
            out[name] = value
    return out


async def load_scenario_from_artifact(
    artifact: GraphArtifact,
    *,
    objective_target: PromptTarget,
    allow_drift: bool = False,
    opaque_materializers: Mapping[str, Callable[[dict[str, Any]], Any]] | None = None,
) -> Scenario:
    """
    Rebuild and initialize a scenario from a :class:`GraphArtifact`.

    Args:
        artifact: The artifact (typically loaded via :func:`graph_artifact_from_yaml`).
        objective_target: The target to run the scenario against. NOT captured
            in the artifact (it's environment-specific and frequently opaque);
            always required at load time.
        allow_drift: When ``True``, version + topology-hash mismatches are
            logged but not fatal. Default ``False`` mirrors the strict-fail
            resume contract on :class:`Scenario`.
        opaque_materializers: Per-role-name callables for rebuilding opaque
            ``init_inputs`` from their stored identifier payloads. See
            :func:`materialize_opaque_inputs`.

    Returns:
        Scenario: A fully initialized scenario equivalent (modulo drift) to the
            one that produced the artifact.

    Raises:
        GraphArtifactSecurityError: If the captured FQN is not registry-whitelisted.
        GraphArtifactDriftError: On version or topology-hash mismatch when
            ``allow_drift=False``.
        OpaqueInputUnresolvedError: If an opaque role has no materializer.
    """
    cls = _resolve_scenario_fqn(artifact.scenario_class_fqn)

    if cls.__name__ != artifact.scenario_class_fqn.rsplit(".", 1)[-1] and not allow_drift:
        raise GraphArtifactDriftError(
            f"Class name mismatch: artifact claims {artifact.scenario_class_fqn!r} but resolved {cls.__name__!r}."
        )

    materialized_init_inputs = materialize_opaque_inputs(
        cls,
        artifact.init_inputs,
        opaque_materializers=opaque_materializers,
    )

    strategy_cls = cls.get_strategy_class()
    rebuilt_strategies = [strategy_cls[name] for name in artifact.scenario_strategies]
    dataset_config = _deserialize_dataset_config(artifact.dataset_config)

    init_async = dict(artifact.init_async_inputs)
    init_async.setdefault("objective_target", objective_target)
    init_async.setdefault("scenario_strategies", rebuilt_strategies)
    if dataset_config is not None:
        init_async.setdefault("dataset_config", dataset_config)
    init_async.setdefault("include_baseline", artifact.include_baseline)
    if artifact.memory_labels:
        init_async.setdefault("memory_labels", artifact.memory_labels)

    scenario = await build_scenario_from_inputs(
        cls,
        init_inputs=materialized_init_inputs,
        init_async_inputs=init_async,
    )

    rebuilt_summary = build_topology_summary(scenario)
    rebuilt_hash = _topology_hash(rebuilt_summary)
    if rebuilt_hash != artifact.topology_hash and not allow_drift:
        raise GraphArtifactDriftError(
            f"Topology hash mismatch after rebuild. "
            f"Artifact: {artifact.topology_hash}, rebuilt: {rebuilt_hash}. "
            "Set allow_drift=True to bypass."
        )

    return scenario
