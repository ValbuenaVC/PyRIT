# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from pyrit.exceptions import PluginDiscoveryError
from pyrit.setup import PluginFormat, PluginSpec
from pyrit.setup.plugin_discovery import (
    discover_scenarios,
    discover_techniques,
    import_plugin_async,
)
from pyrit.setup.plugin_formats import SourcePluginFormat


@pytest.fixture
def restore_import_state() -> Iterator[None]:
    path_snapshot = list(sys.path)
    module_snapshot = set(sys.modules)
    yield
    sys.path[:] = path_snapshot
    for name in set(sys.modules) - module_snapshot:
        del sys.modules[name]


async def _prepare_and_import_source(source: Path, *, name: str):
    spec = PluginSpec(name=name, format=PluginFormat.SOURCE, source=source)
    prepared = await SourcePluginFormat().prepare_async(spec=spec)
    return await import_plugin_async(prepared=prepared)


@pytest.mark.usefixtures("restore_import_state")
async def test_discover_scenario_from_single_file(tmp_path: Path) -> None:
    source = tmp_path / "private_scenario.py"
    source.write_text(
        """from pyrit.scenario.scenarios.airt.rapid_response import RapidResponse

class PrivateScenario(RapidResponse):
    pass
""",
        encoding="utf-8",
    )

    imported = await _prepare_and_import_source(source, name="private")
    contributions = discover_scenarios(imported=imported)

    assert len(contributions) == 1
    assert contributions[0].scenario_class.__name__ == "PrivateScenario"
    assert contributions[0].registry_name == "private_scenario"


@pytest.mark.usefixtures("restore_import_state")
async def test_discover_scenario_ignores_imported_foreign_class(tmp_path: Path) -> None:
    source = tmp_path / "foreign_only.py"
    source.write_text(
        "from pyrit.scenario.scenarios.airt.rapid_response import RapidResponse\n",
        encoding="utf-8",
    )

    imported = await _prepare_and_import_source(source, name="foreign_only")

    assert discover_scenarios(imported=imported) == []


@pytest.mark.usefixtures("restore_import_state")
async def test_discover_scenario_rejects_multiple_classes_in_one_module(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous.py"
    source.write_text(
        """from pyrit.scenario.scenarios.airt.rapid_response import RapidResponse

class FirstScenario(RapidResponse):
    pass

class SecondScenario(RapidResponse):
    pass
""",
        encoding="utf-8",
    )

    imported = await _prepare_and_import_source(source, name="ambiguous")

    with pytest.raises(PluginDiscoveryError, match="multiple"):
        discover_scenarios(imported=imported)


@pytest.mark.usefixtures("restore_import_state")
async def test_discover_scenario_uses_source_relative_dotted_name(tmp_path: Path) -> None:
    package = tmp_path / "operation_plugin"
    scenarios = package / "scenarios" / "image"
    scenarios.mkdir(parents=True)
    for init_file in (package / "__init__.py", package / "scenarios" / "__init__.py", scenarios / "__init__.py"):
        init_file.write_text("", encoding="utf-8")
    (scenarios / "abuse.py").write_text(
        """from pyrit.scenario.scenarios.airt.rapid_response import RapidResponse

class AbuseScenario(RapidResponse):
    pass
""",
        encoding="utf-8",
    )

    imported = await _prepare_and_import_source(package, name="operation")
    contributions = discover_scenarios(imported=imported)

    assert [item.registry_name for item in contributions] == ["image.abuse"]


@pytest.mark.usefixtures("restore_import_state")
async def test_discover_technique_from_conventional_factory_function(tmp_path: Path) -> None:
    source = tmp_path / "private_technique.py"
    source.write_text(
        """from pyrit.executor.attack import PromptSendingAttack
from pyrit.scenario import AttackTechniqueFactory

def get_technique_factories():
    return [
        AttackTechniqueFactory(
            name="operation_foobar",
            attack_class=PromptSendingAttack,
            strategy_tags=["single_turn", "scenario:airt.rapid_response"],
        )
    ]
""",
        encoding="utf-8",
    )

    imported = await _prepare_and_import_source(source, name="private_technique")
    contributions = discover_techniques(imported=imported)

    assert len(contributions) == 1
    assert contributions[0].factory.name == "operation_foobar"
    assert contributions[0].scenario_names == frozenset({"airt.rapid_response"})


@pytest.mark.usefixtures("restore_import_state")
async def test_discover_technique_from_explicit_contribution(tmp_path: Path) -> None:
    source = tmp_path / "explicit_technique.py"
    source.write_text(
        """from pyrit.executor.attack import PromptSendingAttack
from pyrit.scenario import AttackTechniqueFactory
from pyrit.setup.plugin_discovery import TechniqueContribution

OPERATION = TechniqueContribution(
    factory=AttackTechniqueFactory(
        name="operation_explicit",
        attack_class=PromptSendingAttack,
        strategy_tags=["single_turn"],
    ),
    scenario_names=frozenset({"airt.rapid_response"}),
)
""",
        encoding="utf-8",
    )

    imported = await _prepare_and_import_source(source, name="explicit_technique")
    contributions = discover_techniques(imported=imported)

    assert [item.factory.name for item in contributions] == ["operation_explicit"]


@pytest.mark.usefixtures("restore_import_state")
async def test_discover_technique_from_module_global_factory(tmp_path: Path) -> None:
    source = tmp_path / "global_technique.py"
    source.write_text(
        """from pyrit.executor.attack import PromptSendingAttack
from pyrit.scenario import AttackTechniqueFactory

OPERATION = AttackTechniqueFactory(
    name="operation_global",
    attack_class=PromptSendingAttack,
    strategy_tags=["single_turn", "scenario:airt.rapid_response"],
)
""",
        encoding="utf-8",
    )

    imported = await _prepare_and_import_source(source, name="global_technique")
    contributions = discover_techniques(imported=imported)

    assert [item.factory.name for item in contributions] == ["operation_global"]


@pytest.mark.usefixtures("restore_import_state")
async def test_discover_technique_does_not_call_arbitrary_helpers(tmp_path: Path) -> None:
    source = tmp_path / "safe_discovery.py"
    marker = tmp_path / "helper_called.txt"
    source.write_text(
        f"""from pathlib import Path
from pyrit.executor.attack import PromptSendingAttack
from pyrit.scenario import AttackTechniqueFactory

def build_something():
    Path({str(marker)!r}).write_text("called")
    return AttackTechniqueFactory(name="hidden", attack_class=PromptSendingAttack)

OPERATION = AttackTechniqueFactory(
    name="visible",
    attack_class=PromptSendingAttack,
    strategy_tags=["scenario:airt.rapid_response"],
)
""",
        encoding="utf-8",
    )

    imported = await _prepare_and_import_source(source, name="safe_discovery")
    contributions = discover_techniques(imported=imported)

    assert [item.factory.name for item in contributions] == ["visible"]
    assert not marker.exists()


@pytest.mark.usefixtures("restore_import_state")
async def test_discover_technique_requires_scenario_applicability(tmp_path: Path) -> None:
    source = tmp_path / "missing_applicability.py"
    source.write_text(
        """from pyrit.executor.attack import PromptSendingAttack
from pyrit.scenario import AttackTechniqueFactory

OPERATION = AttackTechniqueFactory(name="missing_scope", attack_class=PromptSendingAttack)
""",
        encoding="utf-8",
    )

    imported = await _prepare_and_import_source(source, name="missing_applicability")

    with pytest.raises(PluginDiscoveryError, match="applicable scenario"):
        discover_techniques(imported=imported)


@pytest.mark.usefixtures("restore_import_state")
async def test_discover_technique_rejects_duplicate_names(tmp_path: Path) -> None:
    source = tmp_path / "duplicate_techniques.py"
    source.write_text(
        """from pyrit.executor.attack import PromptSendingAttack
from pyrit.scenario import AttackTechniqueFactory

FIRST = AttackTechniqueFactory(
    name="duplicate",
    attack_class=PromptSendingAttack,
    strategy_tags=["scenario:airt.rapid_response"],
)
SECOND = AttackTechniqueFactory(
    name="duplicate",
    attack_class=PromptSendingAttack,
    strategy_tags=["scenario:airt.rapid_response"],
)
""",
        encoding="utf-8",
    )

    imported = await _prepare_and_import_source(source, name="duplicate_techniques")

    with pytest.raises(PluginDiscoveryError, match="duplicate"):
        discover_techniques(imported=imported)
