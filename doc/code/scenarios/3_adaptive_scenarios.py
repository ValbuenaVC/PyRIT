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
# # Adaptive Scenarios
#
# An **adaptive scenario** doesn't run every attack technique against every objective.
# Instead, it picks which technique to try next per-objective, learns from what worked,
# and stops as soon as one technique succeeds. This concentrates spend on techniques
# that actually work on your target.
#
# ## How it works (high level)
#
# For each objective, the scenario tries up to `max_attempts_per_objective` techniques:
#
# - With probability `epsilon`, it **explores** — picks a random technique.
# - Otherwise it **exploits** — picks the technique with the highest observed success
#   rate so far.
# - It records the outcome and stops early on success.
#
# Unseen techniques are tried first, so the first few objectives effectively round-robin
# through every technique before the scenario settles on the best performers.
#
# ## Adaptive vs. static scenarios
#
# | Feature             | Static scenarios                  | Adaptive scenarios                 |
# |---------------------|-----------------------------------|------------------------------------|
# | Technique selection | Run every selected technique      | Pick per-objective from outcomes   |
# | Early stopping      | No                                | Yes — stops on first success       |
# | Cost                | O(techniques × objectives)        | O(max_attempts × objectives)       |
#
# `AdaptiveScenario` is the modality-agnostic base class.
# [`TextAdaptive`](../../../pyrit/scenario/scenarios/adaptive/text_adaptive.py) is the
# text subclass used in the examples below.

# %% [markdown]
# ## Setup

# %%
from pathlib import Path

from pyrit.registry import TargetRegistry
from pyrit.scenario import DatasetConfiguration
from pyrit.scenario.printer.console_printer import ConsoleScenarioResultPrinter
from pyrit.scenario.scenarios.adaptive import TextAdaptive, harm_category_context
from pyrit.setup import initialize_from_config_async

await initialize_from_config_async(config_path=Path("../../scanner/pyrit_conf.yaml"))  # type: ignore

objective_target = TargetRegistry.get_registry_singleton().get_instance_by_name("openai_chat")
printer = ConsoleScenarioResultPrinter()

# %% [markdown]
# ## Basic usage
#
# Defaults: `epsilon=0.2`, `max_attempts_per_objective=3`, the subclass's default datasets.

# %%
scenario = TextAdaptive()

await scenario.initialize_async(  # type: ignore
    objective_target=objective_target,
)
result = await scenario.run_async()  # type: ignore
await printer.write_async(result)  # type: ignore

# %% [markdown]
# ## Configuring a run
#
# All the knobs below are constructor or `initialize_async` arguments — combine whichever
# you need on a single scenario instance:
#
# - **`epsilon`** — exploration probability. `0.0` is pure exploit, `1.0` is pure random,
#   `0.2` (default) is 20% exploration.
# - **`max_attempts_per_objective`** — caps techniques tried per objective. Higher means
#   more chances to succeed and more API calls.
# - **`context_extractor`** — partitions the success-rate table. The default
#   `global_context` keeps one shared table; `harm_category_context` learns each harm
#   category independently. Custom callables of type `Callable[[SeedAttackGroup], str]`
#   are supported.
# - **`seed`** — makes every selection decision deterministic.
# - **`scenario_strategies`** (on `initialize_async`) — restricts which techniques the
#   selector can pick from. Use `TextAdaptive.get_strategy_class()` to access the enum.
#
# The cell below exercises all of them at once.

# %%
strategy_class = TextAdaptive.get_strategy_class()

configured_scenario = TextAdaptive(
    epsilon=0.3,
    max_attempts_per_objective=5,
    context_extractor=harm_category_context,
    seed=42,
)

await configured_scenario.initialize_async(  # type: ignore
    objective_target=objective_target,
    scenario_strategies=[strategy_class("single_turn")],
    dataset_config=DatasetConfiguration(
        dataset_names=["airt_hate", "airt_violence"],
        max_dataset_size=4,
    ),
)
configured_result = await configured_scenario.run_async()  # type: ignore
await printer.write_async(configured_result)  # type: ignore

# %% [markdown]
# ## Resuming a run
#
# Adaptive scenarios are resumable — pass `scenario_result_id=...` to the `TextAdaptive`
# constructor and the run picks up where it left off, with prior outcomes replayed into
# the selector. Resume must use the same configuration as the original run.

# %%
resumed_scenario = TextAdaptive(
    epsilon=0.3,
    max_attempts_per_objective=5,
    context_extractor=harm_category_context,
    seed=42,
    scenario_result_id=str(configured_result.id),
)

await resumed_scenario.initialize_async(  # type: ignore
    objective_target=objective_target,
    scenario_strategies=[strategy_class("single_turn")],
    dataset_config=DatasetConfiguration(
        dataset_names=["airt_hate", "airt_violence"],
        max_dataset_size=4,
    ),
)
resumed_result = await resumed_scenario.run_async()  # type: ignore
await printer.write_async(resumed_result)  # type: ignore

# %% [markdown]
# ## Inspecting which techniques were tried
#
# The dispatcher stamps every objective's `AttackResult.metadata` with:
#
# - `adaptive_context` — the bucket key from the `context_extractor`.
# - `adaptive_attempts` — the ordered list of `{"technique", "outcome"}` dicts
#   recording exactly which techniques the selector picked and what happened.
#
# Walk that metadata to see the per-objective trail and aggregate counts.

# %%
from collections import Counter

# Per-objective trail
for results in resumed_result.attack_results.values():
    for r in results:
        attempts = r.metadata.get("adaptive_attempts", [])
        trail = " → ".join(f"{a['technique']}({a['outcome']})" for a in attempts)
        print(f"[{r.outcome.value:7s}] {r.objective!r}: {trail}")

# Aggregate per-technique pick counts and success rate across the run
picks: Counter[str] = Counter()
wins: Counter[str] = Counter()
for results in resumed_result.attack_results.values():
    for r in results:
        for step in r.metadata.get("adaptive_attempts", []):
            picks[step["technique"]] += 1
            if step["outcome"] == "success":
                wins[step["technique"]] += 1

print("\nTechnique             wins / picks   rate")
for technique, n in picks.most_common():
    print(f"{technique:20s}  {wins[technique]:>4} / {n:<4}   {wins[technique] / n:.0%}")
