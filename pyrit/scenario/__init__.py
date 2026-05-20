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
from pyrit.scenario.core import (
    AtomicAttack,
    AttackTechnique,
    AttackTechniqueFactory,
    BaselineAttackPolicy,
    DatasetConfiguration,
    PolicyAction,
    RoleDescriptor,
    RoleTag,
    Scenario,
    ScenarioCompositeStrategy,
    ScenarioCoreState,
    ScenarioStateLike,
    ScenarioStep,
    ScenarioStepResult,
    ScenarioStrategy,
    StrategyGraph,
    StrategyPolicy,
    linear_strategy_policy,
    policy_to_spec,
    spec_to_enum,
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

__all__ = [
    "AtomicAttack",
    "AttackTechnique",
    "AttackTechniqueFactory",
    "BaselineAttackPolicy",
    "DatasetConfiguration",
    "Parameter",
    "PolicyAction",
    "RoleDescriptor",
    "RoleTag",
    "Scenario",
    "ScenarioCompositeStrategy",
    "ScenarioCoreState",
    "ScenarioStateLike",
    "ScenarioStep",
    "ScenarioStepResult",
    "ScenarioStrategy",
    "ScenarioIdentifier",
    "ScenarioResult",
    "StrategyGraph",
    "StrategyPolicy",
    "adaptive",
    "airt",
    "benchmark",
    "foundry",
    "garak",
    "linear_strategy_policy",
    "policy_to_spec",
    "spec_to_enum",
]
