# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from pyrit.datasets import SeedDatasetProvider
from pyrit.exceptions import PluginCollisionError, PluginValidationError
from pyrit.registry import AttackTechniqueRegistry, ScenarioRegistry
from pyrit.setup import PluginFormat, PluginSpec
from pyrit.setup.plugin_loader import PluginInitializer


@pytest.fixture(autouse=True)
def reset_plugin_registration_state() -> Iterator[None]:
    path_snapshot = list(sys.path)
    module_snapshot = set(sys.modules)
    provider_snapshot = dict(SeedDatasetProvider._registry)
    AttackTechniqueRegistry.reset_registry_singleton()
    ScenarioRegistry.reset_registry_singleton()
    yield
    AttackTechniqueRegistry.reset_registry_singleton()
    ScenarioRegistry.reset_registry_singleton()
    SeedDatasetProvider._registry.clear()
    SeedDatasetProvider._registry.update(provider_snapshot)
    sys.path[:] = path_snapshot
    for name in set(sys.modules) - module_snapshot:
        del sys.modules[name]


def _source_spec(source: Path, *, name: str = "operation") -> PluginSpec:
    return PluginSpec(name=name, format=PluginFormat.SOURCE, source=source)


async def test_plugin_initializer_registers_source_technique(tmp_path: Path) -> None:
    source = tmp_path / "operation_technique.py"
    source.write_text(
        """from pyrit.executor.attack import PromptSendingAttack
from pyrit.scenario import AttackTechniqueFactory

OPERATION = AttackTechniqueFactory(
    name="operation_foobar",
    attack_class=PromptSendingAttack,
    strategy_tags=["single_turn", "scenario:airt.rapid_response"],
)
""",
        encoding="utf-8",
    )

    await PluginInitializer(plugins=[_source_spec(source)]).initialize_async()

    registry = AttackTechniqueRegistry.get_registry_singleton()
    assert registry.get_factories()["operation_foobar"] is not None
    entry = registry.instances.get_entry("operation_foobar")
    assert entry is not None
    assert entry.metadata["plugin_name"] == "operation"
    assert entry.metadata["scenario_names"] == ["airt.rapid_response"]


async def test_plugin_initializer_registers_source_scenario(
    tmp_path: Path,
    patch_central_database: None,
) -> None:
    source = tmp_path / "private_scenario.py"
    source.write_text(
        """from pyrit.scenario.scenarios.airt.rapid_response import RapidResponse

class PrivateScenario(RapidResponse):
    pass
""",
        encoding="utf-8",
    )

    with patch.dict(
        "os.environ",
        {
            "OPENAI_CHAT_ENDPOINT": "https://example.test",
            "OPENAI_CHAT_KEY": "test-key",
            "OPENAI_CHAT_MODEL": "test-model",
        },
    ):
        await PluginInitializer(plugins=[_source_spec(source)]).initialize_async()

    scenario_class = ScenarioRegistry.get_registry_singleton().get_class("private_scenario")
    assert scenario_class.__name__ == "PrivateScenario"


async def test_plugin_initializer_rejects_latent_builtin_technique_name(tmp_path: Path) -> None:
    source = tmp_path / "collision.py"
    source.write_text(
        """from pyrit.executor.attack import PromptSendingAttack
from pyrit.scenario import AttackTechniqueFactory

OPERATION = AttackTechniqueFactory(
    name="role_play",
    attack_class=PromptSendingAttack,
    strategy_tags=["scenario:airt.rapid_response"],
)
""",
        encoding="utf-8",
    )

    with pytest.raises(PluginCollisionError, match="role_play"):
        await PluginInitializer(plugins=[_source_spec(source)]).initialize_async()

    assert "role_play" not in AttackTechniqueRegistry.get_registry_singleton().get_factories()


async def test_plugin_initializer_rolls_back_scenario_when_technique_collides(tmp_path: Path) -> None:
    source = tmp_path / "partial.py"
    source.write_text(
        """from pyrit.executor.attack import PromptSendingAttack
from pyrit.scenario import AttackTechniqueFactory
from pyrit.scenario.scenarios.airt.rapid_response import RapidResponse

class PartialScenario(RapidResponse):
    pass

OPERATION = AttackTechniqueFactory(
    name="role_play",
    attack_class=PromptSendingAttack,
    strategy_tags=["scenario:airt.rapid_response"],
)
""",
        encoding="utf-8",
    )

    with pytest.raises(PluginCollisionError):
        await PluginInitializer(plugins=[_source_spec(source)]).initialize_async()

    assert "partial" not in ScenarioRegistry.get_registry_singleton()._classes
    assert str(tmp_path.resolve()) not in sys.path
    assert "partial" not in sys.modules


async def test_plugin_initializer_removes_unsupported_provider_side_effect(tmp_path: Path) -> None:
    source = tmp_path / "provider_side_effect.py"
    source.write_text(
        """from pyrit.datasets import SeedDatasetProvider
from pyrit.executor.attack import PromptSendingAttack
from pyrit.models import SeedDataset
from pyrit.scenario import AttackTechniqueFactory

class PrivateProvider(SeedDatasetProvider):
    @property
    def dataset_name(self):
        return "private"

    async def fetch_dataset_async(self, *, cache=True):
        return SeedDataset(seeds=[], dataset_name="private")

OPERATION = AttackTechniqueFactory(
    name="provider_safe",
    attack_class=PromptSendingAttack,
    strategy_tags=["scenario:airt.rapid_response"],
)
""",
        encoding="utf-8",
    )

    await PluginInitializer(plugins=[_source_spec(source)]).initialize_async()

    assert "PrivateProvider" not in SeedDatasetProvider.get_all_providers()


async def test_plugin_initializer_rejects_scenario_that_cannot_build_metadata(tmp_path: Path) -> None:
    source = tmp_path / "invalid_scenario.py"
    source.write_text(
        """from pyrit.scenario.scenarios.airt.rapid_response import RapidResponse

class InvalidScenario(RapidResponse):
    def __init__(self, *, required_value):
        super().__init__()
""",
        encoding="utf-8",
    )

    with pytest.raises(PluginValidationError, match="metadata"):
        await PluginInitializer(plugins=[_source_spec(source)]).initialize_async()

    assert "invalid_scenario" not in ScenarioRegistry.get_registry_singleton()._classes
