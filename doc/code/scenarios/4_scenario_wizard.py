# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # The Scenario Wizard
#
# The **wizard** is a thin shell around three lower-level surfaces that make scenarios
# composable from inputs rather than hand-written code:
#
# 1. **`Scenario.input_schema()`** — declares the rich-object inputs a scenario's
#    `__init__` takes (targets, scorers, opaque steps).
# 2. **`Scenario.supported_parameters()`** — declares the scalar inputs
#    `initialize_async` takes (dataset config, strategies, concurrency).
# 3. **`build_scenario_from_inputs`** — drives both phases from a single dict so the
#    same call works from a CLI prompt, a Jupyter notebook, or a saved artifact.
#
# This notebook walks through the wizard from the inside: discovering schemas,
# building a scenario programmatically, capturing it as a **graph artifact** for
# later replay, and reloading the artifact. The interactive CLI (`pyrit_wizard`)
# and the replay command (`pyrit_scan --from-artifact`) are documented at the end.

# %% [markdown]
# ## Setup

# %%
from pathlib import Path
from tempfile import TemporaryDirectory

from pyrit.registry import ScenarioRegistry, TargetRegistry
from pyrit.scenario.core import (
    build_graph_artifact,
    build_scenario_from_inputs,
    discover_input_schema,
    discover_supported_parameters,
    graph_artifact_from_yaml,
    graph_artifact_to_yaml,
    load_scenario_from_artifact,
)
from pyrit.scenario.scenarios.adaptive import TextAdaptive
from pyrit.setup import initialize_from_config_async

await initialize_from_config_async(config_path=Path("../../scanner/pyrit_conf.yaml"))  # type: ignore

objective_target = TargetRegistry.get_registry_singleton().get_instance_by_name("openai_chat")

# %% [markdown]
# ## 1. Discovering what a scenario needs
#
# Before building, ask the scenario class what it expects. The two schemas are
# orthogonal: `input_schema()` covers constructor arguments, `supported_parameters()`
# covers `initialize_async`. Every `Scenario` subclass exposes both — most inherit
# the default empty `input_schema()` and only declare scalars in `supported_parameters`.

# %%
print("TextAdaptive constructor inputs (input_schema):")
for role in discover_input_schema(TextAdaptive):
    required = "required" if role.required else "optional"
    default = f"default={role.default!r}" if role.default is not None else ""
    print(f"  - {role.name:32s} {role.tag.value:13s} {required:9s} {default}")

print("\nTextAdaptive initialize_async inputs (supported_parameters):")
for param in discover_supported_parameters(TextAdaptive):
    print(f"  - {param.name:32s} default={param.default!r}")

# %% [markdown]
# All four `TextAdaptive` input-schema roles are optional with defaults, so the
# wizard can build the scenario from a completely empty input dict. Scenarios that
# declare required `OPAQUE` roles (like `BroadSweepThenDeepDive`) cannot be built
# this way — the CLI wizard rejects them with a hint to use `--from-artifact`,
# and programmatic callers must supply the rich objects directly.

# %% [markdown]
# ## 2. Building a scenario from inputs
#
# `build_scenario_from_inputs` constructs the scenario, then runs `initialize_async`.
# It returns a fully initialized scenario ready for `run_async`. The same call shape
# works from any front-end — what changes is who provides the input dicts.

# %%
strategy_class = TextAdaptive.get_strategy_class()

scenario = await build_scenario_from_inputs(  # type: ignore
    TextAdaptive,
    init_inputs={
        "epsilon": 0.3,
        "max_attempts_per_objective": 4,
        "seed": 42,
    },
    init_async_inputs={
        "objective_target": objective_target,
        "scenario_strategies": [strategy_class("single_turn")],
    },
)

print(f"Built scenario: {scenario.name}")
print(f"  scenario_strategies: {[s.value for s in scenario._scenario_strategies]}")
print(f"  epsilon: {scenario._epsilon}, max_attempts: {scenario._max_attempts_per_objective}")

# %% [markdown]
# ## 3. Capturing the scenario as a graph artifact
#
# A `GraphArtifact` is a frozen snapshot of everything needed to rebuild the same
# scenario later: the class FQN, the constructor inputs (opaque rich objects encoded
# by `ComponentIdentifier`), the `initialize_async` inputs, and a topology hash for
# drift detection. The objective target is **not** captured — it's environment-specific.

# %%
artifact = build_graph_artifact(scenario)
print(f"  scenario_class_fqn:  {artifact.scenario_class_fqn}")
print(f"  scenario_version:    {artifact.scenario_version}")
print(f"  pyrit_version:       {artifact.pyrit_version}")
print(f"  topology_hash:       {artifact.topology_hash[:12]}…")
print(f"  init_async_inputs:   {sorted(artifact.init_async_inputs.keys())}")

# %% [markdown]
# Serialize to YAML for sharing or version control. The dump is canonical (sorted
# keys, block-style) so equivalent scenarios produce byte-identical artifacts.

# %%
with TemporaryDirectory() as tmpdir:
    artifact_path = Path(tmpdir) / "text_adaptive.yaml"
    graph_artifact_to_yaml(artifact, artifact_path)
    print(artifact_path.read_text()[:800])

# %% [markdown]
# ## 4. Reloading the artifact
#
# `load_scenario_from_artifact` re-runs the registered scenario class through
# `build_scenario_from_inputs` with the captured inputs, then verifies the
# rebuilt graph's topology hash matches the captured one. Drift fails by default;
# pass `allow_drift=True` to tolerate version or topology changes.

# %%
with TemporaryDirectory() as tmpdir:
    artifact_path = Path(tmpdir) / "text_adaptive.yaml"
    graph_artifact_to_yaml(artifact, artifact_path)

    reloaded_artifact = graph_artifact_from_yaml(artifact_path)
    reloaded_scenario = await load_scenario_from_artifact(  # type: ignore
        reloaded_artifact,
        objective_target=objective_target,
    )

print(f"Reloaded scenario: {reloaded_scenario.name}")
print(f"  topology_hash matches: {build_graph_artifact(reloaded_scenario).topology_hash == artifact.topology_hash}")

# %% [markdown]
# ## 5. The interactive CLI surface
#
# Two CLIs sit on top of the building blocks above:
#
# - **`pyrit_wizard`** — prompts for each role declared by the chosen scenario's
#   `input_schema()` and `supported_parameters()`, then either runs the scenario
#   (`--run`) or persists it as a graph artifact (`--save path.yaml`). Useful for
#   first-time exploration or quick one-offs.
#
# - **`pyrit_scan --from-artifact path.yaml --target <name>`** — loads a previously
#   saved artifact and replays it against the named target. Use this in CI or to
#   share a reproducible attack with a collaborator. `--allow-drift` tolerates
#   scenario_version or topology-hash mismatches.
#
# The wizard cannot elicit `BroadSweepThenDeepDive` (or any scenario with
# required `OPAQUE` roles) directly — its constructor takes pre-built
# `AtomicAttack` instances and closures the CLI cannot construct. For those
# scenarios, build once programmatically (as in section 2 above), save the
# artifact, and replay via `pyrit_scan --from-artifact`.

# %% [markdown]
# ## Discovery: what scenarios are available?
#
# Scenarios are self-registering via `ScenarioRegistry`. The wizard's `--list`
# flag delegates to the same metadata.

# %%
scenario_registry = ScenarioRegistry.get_registry_singleton()
for metadata in sorted(scenario_registry.list_metadata(), key=lambda m: m.registry_name)[:8]:
    summary = metadata.class_description.splitlines()[0] if metadata.class_description else ""
    print(f"  {metadata.registry_name:36s} {summary[:60]}")
