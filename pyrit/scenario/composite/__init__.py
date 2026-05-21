# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Composite scenarios — Scenario instances that compose other scenarios.

The first composite ships in this subpackage is :class:`ScenarioPipeline`, a
linear sequential composer that runs a list of inner scenarios as phases. It
is the direct response to the review suggestion that the simple cross-scenario
composition case should not require authoring a custom branching policy.

Composite scenarios use :class:`StrategyGraph` internally for orchestration so
they remain consistent with the rest of the scenario core, but their public
API surface (:class:`PhaseSpec`, :class:`ConditionalPhaseSpec`) keeps the
state-machine vocabulary hidden from the caller.
"""

from pyrit.scenario.composite.scenario_pipeline import (
    ConditionalPhaseSpec,
    PhaseExecution,
    PhaseSpec,
    PipelineContext,
    ScenarioPipeline,
    ScenarioPipelineStrategy,
)

__all__ = [
    "ConditionalPhaseSpec",
    "PhaseExecution",
    "PhaseSpec",
    "PipelineContext",
    "ScenarioPipeline",
    "ScenarioPipelineStrategy",
]
