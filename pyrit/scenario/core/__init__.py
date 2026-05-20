# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Core scenario classes for running attack configurations."""

from pyrit.common.parameter import Parameter
from pyrit.scenario.core.atomic_attack import AtomicAttack
from pyrit.scenario.core.attack_technique import AttackTechnique
from pyrit.scenario.core.attack_technique_factory import AttackTechniqueFactory, ScorerOverridePolicy
from pyrit.scenario.core.builder import (
    ScenarioInputValidationError,
    build_scenario_from_inputs,
    discover_input_schema,
    discover_supported_parameters,
    validate_init_inputs,
)
from pyrit.scenario.core.dataset_configuration import EXPLICIT_SEED_GROUPS_KEY, DatasetConfiguration
from pyrit.scenario.core.graph_artifact import (
    GraphArtifact,
    GraphArtifactDriftError,
    GraphArtifactError,
    GraphArtifactSecurityError,
    OpaqueInputUnresolvedError,
    build_graph_artifact,
    build_topology_summary,
    graph_artifact_from_yaml,
    graph_artifact_to_yaml,
    load_scenario_from_artifact,
    materialize_opaque_inputs,
)
from pyrit.scenario.core.input_collector import (
    ArtifactInputCollector,
    CliInputCollector,
    DictInputCollector,
    InputCollector,
    MaxAttemptsExceededError,
    OpaqueRoleNotElicitableError,
    collect_inputs_with_retry,
)
from pyrit.scenario.core.input_schema import RoleDescriptor, RoleTag
from pyrit.scenario.core.scenario import BaselineAttackPolicy, Scenario
from pyrit.scenario.core.scenario_state import ScenarioCoreState, ScenarioStateLike
from pyrit.scenario.core.scenario_step import ScenarioStep, ScenarioStepResult
from pyrit.scenario.core.scenario_strategy import ScenarioCompositeStrategy, ScenarioStrategy
from pyrit.scenario.core.scenario_target_defaults import get_default_adversarial_target, get_default_scorer_target
from pyrit.scenario.core.scenario_techniques import (
    SCENARIO_TECHNIQUES,
    register_scenario_techniques,
)
from pyrit.scenario.core.strategy_graph import PolicyAction, StrategyGraph, StrategyPolicy, linear_strategy_policy
from pyrit.scenario.core.waterfall import policy_to_spec, spec_to_enum

__all__ = [
    "ArtifactInputCollector",
    "AtomicAttack",
    "AttackTechnique",
    "AttackTechniqueFactory",
    "BaselineAttackPolicy",
    "CliInputCollector",
    "DatasetConfiguration",
    "DictInputCollector",
    "EXPLICIT_SEED_GROUPS_KEY",
    "GraphArtifact",
    "GraphArtifactDriftError",
    "GraphArtifactError",
    "GraphArtifactSecurityError",
    "InputCollector",
    "MaxAttemptsExceededError",
    "OpaqueInputUnresolvedError",
    "OpaqueRoleNotElicitableError",
    "Parameter",
    "PolicyAction",
    "RoleDescriptor",
    "RoleTag",
    "SCENARIO_TECHNIQUES",
    "Scenario",
    "ScenarioCompositeStrategy",
    "ScenarioCoreState",
    "ScenarioInputValidationError",
    "ScenarioStateLike",
    "ScenarioStep",
    "ScenarioStepResult",
    "ScenarioStrategy",
    "ScorerOverridePolicy",
    "StrategyGraph",
    "StrategyPolicy",
    "build_graph_artifact",
    "build_scenario_from_inputs",
    "build_topology_summary",
    "collect_inputs_with_retry",
    "discover_input_schema",
    "discover_supported_parameters",
    "get_default_adversarial_target",
    "get_default_scorer_target",
    "graph_artifact_from_yaml",
    "graph_artifact_to_yaml",
    "linear_strategy_policy",
    "load_scenario_from_artifact",
    "materialize_opaque_inputs",
    "policy_to_spec",
    "register_scenario_techniques",
    "spec_to_enum",
    "validate_init_inputs",
]
