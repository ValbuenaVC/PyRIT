# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
High-level scenario classes for running attack configurations.

Core classes can be imported directly from this module:
    from pyrit.scenario import Scenario, AtomicAttack, ScenarioStrategy

Specific scenarios should be imported from their subpackages:
    from pyrit.scenario.airt import RapidResponse, Cyber
    from pyrit.scenario.garak import Encoding
    from pyrit.scenario.foundry import RedTeamAgent
"""

import sys

from pyrit.common.parameter import Parameter
from pyrit.models.scenario_result import ScenarioIdentifier, ScenarioResult
from pyrit.scenario import composite as _composite_module
from pyrit.scenario.composite import (
    ConditionalPhaseSpec,
    PhaseExecution,
    PhaseSpec,
    PipelineContext,
    ScenarioPipeline,
    ScenarioPipelineStrategy,
)
from pyrit.scenario.core import (
    ArtifactInputCollector,
    AtomicAttack,
    AttackTechnique,
    AttackTechniqueFactory,
    BaselineAttackPolicy,
    CliInputCollector,
    DatasetConfiguration,
    DictInputCollector,
    GraphArtifact,
    GraphArtifactDriftError,
    GraphArtifactError,
    GraphArtifactSecurityError,
    InputCollector,
    LinearScenario,
    LinearScenarioStrategy,
    MaxAttemptsExceededError,
    OpaqueInputUnresolvedError,
    OpaqueRoleNotElicitableError,
    PolicyAction,
    RoleDescriptor,
    RoleTag,
    Scenario,
    ScenarioCompositeStrategy,
    ScenarioCoreState,
    ScenarioInputValidationError,
    ScenarioStateLike,
    ScenarioStep,
    ScenarioStepResult,
    ScenarioStrategy,
    StrategyGraph,
    StrategyPolicy,
    build_graph_artifact,
    build_scenario_from_inputs,
    build_topology_summary,
    collect_inputs_with_retry,
    discover_input_schema,
    discover_supported_parameters,
    enum_to_spec,
    graph_artifact_from_yaml,
    graph_artifact_to_yaml,
    linear_strategy_policy,
    load_scenario_from_artifact,
    materialize_opaque_inputs,
    policy_to_spec,
    spec_to_enum,
    spec_to_policy_inputs,
    validate_init_async_inputs,
    validate_init_inputs,
)

# Import scenario submodules directly and register them as virtual subpackages
# This allows: from pyrit.scenario.airt import ContentHarms
# without needing separate pyrit/scenario/airt/ directories
from pyrit.scenario.scenarios import adaptive as _adaptive_module
from pyrit.scenario.scenarios import airt as _airt_module
from pyrit.scenario.scenarios import benchmark as _benchmark_module
from pyrit.scenario.scenarios import foundry as _foundry_module
from pyrit.scenario.scenarios import garak as _garak_module

sys.modules["pyrit.scenario.adaptive"] = _adaptive_module
sys.modules["pyrit.scenario.airt"] = _airt_module
sys.modules["pyrit.scenario.benchmark"] = _benchmark_module
sys.modules["pyrit.scenario.garak"] = _garak_module
sys.modules["pyrit.scenario.foundry"] = _foundry_module

# Also expose as attributes for IDE support
adaptive = _adaptive_module
airt = _airt_module
benchmark = _benchmark_module
garak = _garak_module
foundry = _foundry_module
composite = _composite_module

__all__ = [
    "ArtifactInputCollector",
    "AtomicAttack",
    "AttackTechnique",
    "AttackTechniqueFactory",
    "BaselineAttackPolicy",
    "CliInputCollector",
    "ConditionalPhaseSpec",
    "DatasetConfiguration",
    "DictInputCollector",
    "GraphArtifact",
    "GraphArtifactDriftError",
    "GraphArtifactError",
    "GraphArtifactSecurityError",
    "InputCollector",
    "LinearScenario",
    "LinearScenarioStrategy",
    "MaxAttemptsExceededError",
    "OpaqueInputUnresolvedError",
    "OpaqueRoleNotElicitableError",
    "Parameter",
    "PhaseExecution",
    "PhaseSpec",
    "PipelineContext",
    "PolicyAction",
    "RoleDescriptor",
    "RoleTag",
    "Scenario",
    "ScenarioCompositeStrategy",
    "ScenarioCoreState",
    "ScenarioIdentifier",
    "ScenarioInputValidationError",
    "ScenarioPipeline",
    "ScenarioPipelineStrategy",
    "ScenarioResult",
    "ScenarioStateLike",
    "ScenarioStep",
    "ScenarioStepResult",
    "ScenarioStrategy",
    "StrategyGraph",
    "StrategyPolicy",
    "adaptive",
    "airt",
    "benchmark",
    "build_graph_artifact",
    "build_scenario_from_inputs",
    "build_topology_summary",
    "collect_inputs_with_retry",
    "composite",
    "discover_input_schema",
    "discover_supported_parameters",
    "enum_to_spec",
    "foundry",
    "garak",
    "graph_artifact_from_yaml",
    "graph_artifact_to_yaml",
    "linear_strategy_policy",
    "load_scenario_from_artifact",
    "materialize_opaque_inputs",
    "policy_to_spec",
    "spec_to_enum",
    "spec_to_policy_inputs",
    "validate_init_async_inputs",
    "validate_init_inputs",
]
