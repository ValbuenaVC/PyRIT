# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Integration test for the initializer-pointing plug-in path.

Exercises the full source-plug-in flow against real registries (no mocks): a source
package whose ``PyRITInitializer`` registers a private scenario and a private attack
technique becomes discoverable through the standard registries after the privileged
plug-in phase runs.
"""

import sys
import textwrap
from pathlib import Path

import pytest

from pyrit.registry import AttackTechniqueRegistry, ScenarioRegistry
from pyrit.setup import PluginSpec
from pyrit.setup.plugin_loader import PluginInitializer

_PLUGIN_PACKAGE = "my_redteam_plugin"


def _write_source_plugin(root: Path) -> None:
    """Write a source package whose initializer registers a scenario and a technique."""
    pkg = root / _PLUGIN_PACKAGE
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "scenario.py").write_text(
        textwrap.dedent(
            '''
            from functools import cache

            from pyrit.executor.attack import PromptSendingAttack
            from pyrit.models import SeedObjective
            from pyrit.registry import AttackTechniqueRegistry
            from pyrit.scenario.core import (
                AttackTechniqueFactory,
                DatasetAttackConfiguration,
                Scenario,
            )
            from pyrit.score import SubStringScorer


            @cache
            def _build_private_technique():
                return AttackTechniqueRegistry.build_technique_class_from_factories(
                    class_name="PrivateTechnique",
                    factories=[
                        AttackTechniqueFactory(name="private_prompt_sending", attack_class=PromptSendingAttack)
                    ],
                    aggregate_tags={},
                )


            class PrivateScenario(Scenario):
                """Private scenario for the plug-in integration test."""

                VERSION = 1

                def __init__(self, *, scenario_result_id=None):
                    technique_class = _build_private_technique()
                    super().__init__(
                        version=self.VERSION,
                        technique_class=technique_class,
                        default_technique=technique_class.ALL,
                        default_dataset_config=DatasetAttackConfiguration(
                            seeds=[SeedObjective(value="integration objective")]
                        ),
                        objective_scorer=SubStringScorer(substring="integration"),
                        scenario_result_id=scenario_result_id,
                    )

                async def _build_atomic_attacks_async(self, *, context):
                    return []
            '''
        ),
        encoding="utf-8",
    )
    (pkg / "bootstrap.py").write_text(
        textwrap.dedent(
            f'''
            from pyrit.executor.attack import PromptSendingAttack
            from pyrit.registry import AttackTechniqueRegistry, ScenarioRegistry
            from pyrit.scenario.core import AttackTechniqueFactory
            from pyrit.setup.pyrit_initializer import PyRITInitializer

            from {_PLUGIN_PACKAGE}.scenario import PrivateScenario


            class PrivateInitializer(PyRITInitializer):
                """Register the private scenario and a private attack technique."""

                async def initialize_async(self) -> None:
                    AttackTechniqueRegistry.get_registry_singleton().register_from_factories(
                        [AttackTechniqueFactory(name="operation_foobar", attack_class=PromptSendingAttack)]
                    )
                    ScenarioRegistry.get_registry_singleton().register_class(
                        PrivateScenario, name="my_redteam.private"
                    )
            '''
        ),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _isolate_registries_and_imports():
    original_path = list(sys.path)
    original_modules = set(sys.modules)
    AttackTechniqueRegistry.reset_registry_singleton()
    ScenarioRegistry.reset_registry_singleton()
    yield
    AttackTechniqueRegistry.reset_registry_singleton()
    ScenarioRegistry.reset_registry_singleton()
    sys.path[:] = original_path
    for name in list(sys.modules):
        if name not in original_modules and name.split(".", 1)[0] == _PLUGIN_PACKAGE:
            del sys.modules[name]


@pytest.mark.run_only_if_all_tests
async def test_source_plugin_initializer_registers_real_components(tmp_path: Path) -> None:
    _write_source_plugin(tmp_path)
    spec = PluginSpec(
        name="my_redteam",
        source=tmp_path,
        initializer=f"{_PLUGIN_PACKAGE}.bootstrap.PrivateInitializer",
    )

    await PluginInitializer(plugins=[spec]).initialize_async()

    scenario_registry = ScenarioRegistry.get_registry_singleton()
    assert "my_redteam.private" in scenario_registry.get_class_names()
    assert scenario_registry.get_class("my_redteam.private").__name__ == "PrivateScenario"
    assert "operation_foobar" in AttackTechniqueRegistry.get_registry_singleton().get_factories()


@pytest.mark.run_only_if_all_tests
async def test_source_plugin_missing_initializer_fails_closed(tmp_path: Path) -> None:
    from pyrit.exceptions import PluginLoadError

    _write_source_plugin(tmp_path)
    spec = PluginSpec(
        name="my_redteam",
        source=tmp_path,
        initializer=f"{_PLUGIN_PACKAGE}.bootstrap.DoesNotExist",
    )

    with pytest.raises(PluginLoadError):
        await PluginInitializer(plugins=[spec]).initialize_async()
