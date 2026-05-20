# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Core scenario classes for running attack configurations."""

from pyrit.common.parameter import Parameter
from pyrit.scenario.core.atomic_attack import AtomicAttack
from pyrit.scenario.core.attack_technique import AttackTechnique
from pyrit.scenario.core.attack_technique_factory import AttackTechniqueFactory, ScorerOverridePolicy
from pyrit.scenario.core.dataset_configuration import EXPLICIT_SEED_GROUPS_KEY, DatasetConfiguration
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
    "AtomicAttack",
    "AttackTechnique",
    "AttackTechniqueFactory",
    "BaselineAttackPolicy",
    "DatasetConfiguration",
    "EXPLICIT_SEED_GROUPS_KEY",
    "PolicyAction",
    "RoleDescriptor",
    "RoleTag",
    "SCENARIO_TECHNIQUES",
    "Parameter",
    "Scenario",
    "ScenarioCompositeStrategy",
    "ScenarioCoreState",
    "ScenarioStateLike",
    "ScenarioStep",
    "ScenarioStepResult",
    "ScenarioStrategy",
    "ScorerOverridePolicy",
    "StrategyGraph",
    "StrategyPolicy",
    "linear_strategy_policy",
    "policy_to_spec",
    "register_scenario_techniques",
    "get_default_scorer_target",
    "get_default_adversarial_target",
    "spec_to_enum",
]
