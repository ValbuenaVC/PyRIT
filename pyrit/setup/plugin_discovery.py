# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Import prepared plug-ins and discover owned PyRIT contributions."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import pkgutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pyrit.exceptions import PluginDiscoveryError, PluginImportError
from pyrit.models import class_name_to_snake_case
from pyrit.scenario import Scenario

if TYPE_CHECKING:
    from types import ModuleType

    from pyrit.setup.plugin_formats import PreparedPlugin


@dataclass(frozen=True)
class ImportedPlugin:
    """The imported modules owned by one prepared plug-in."""

    prepared: PreparedPlugin
    modules: tuple[ModuleType, ...]


@dataclass(frozen=True)
class ScenarioContribution:
    """A discovered scenario class and its proposed registry name."""

    scenario_class: type[Scenario]
    registry_name: str


async def import_plugin_async(*, prepared: PreparedPlugin) -> ImportedPlugin:
    """
    Import a prepared plug-in and all submodules under package entry modules.

    Args:
        prepared (PreparedPlugin): The prepared source or wheel import root.

    Returns:
        ImportedPlugin: The imported, ownership-validated modules.

    Raises:
        PluginImportError: If import or ownership validation fails.
    """
    return await asyncio.to_thread(_import_plugin, prepared=prepared)


def _import_plugin(*, prepared: PreparedPlugin) -> ImportedPlugin:
    import_root = str(prepared.import_root)
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

    try:
        for entry_module in prepared.entry_modules:
            module = importlib.import_module(entry_module)
            _verify_module_location(module=module, import_root=prepared.import_root)
            _import_submodules(module=module)
    except Exception as exc:
        raise PluginImportError(
            f"Could not import plug-in '{prepared.spec.name}' from {prepared.import_root}: {exc}"
        ) from exc

    modules = tuple(
        module
        for name, module in sorted(sys.modules.items())
        if module is not None and _name_owned_by_any(name=name, prefixes=prepared.owned_module_prefixes)
    )
    for module in modules:
        _verify_module_location(module=module, import_root=prepared.import_root)
    return ImportedPlugin(prepared=prepared, modules=modules)


def _import_submodules(*, module: ModuleType) -> None:
    module_path = getattr(module, "__path__", None)
    if not module_path:
        return
    for item in pkgutil.walk_packages(module_path, prefix=f"{module.__name__}."):
        importlib.import_module(item.name)


def _verify_module_location(*, module: ModuleType, import_root: Path) -> None:
    root = import_root.resolve()
    raw_locations = list(getattr(module, "__path__", []) or [])
    module_file = getattr(module, "__file__", None)
    if module_file:
        raw_locations.append(module_file)
    locations = [Path(location).resolve() for location in raw_locations if location]
    if locations and not any(location.is_relative_to(root) for location in locations):
        raise ValueError(f"Imported module '{module.__name__}' resolved outside plug-in root {root}.")


def discover_scenarios(*, imported: ImportedPlugin) -> list[ScenarioContribution]:
    """
    Discover concrete, module-owned Scenario subclasses.

    Args:
        imported (ImportedPlugin): The imported plug-in modules.

    Returns:
        list[ScenarioContribution]: Deterministically ordered scenario contributions.

    Raises:
        PluginDiscoveryError: If one module defines multiple implicit scenarios.
    """
    contributions: list[ScenarioContribution] = []
    for module in imported.modules:
        classes = _module_defined_subclasses(module=module, base_class=Scenario)
        if len(classes) > 1:
            raise PluginDiscoveryError(
                f"Plug-in module '{module.__name__}' defines multiple Scenario classes. "
                "Provide explicit registry names through the plug-in manifest."
            )
        if classes:
            contributions.append(
                ScenarioContribution(
                    scenario_class=classes[0],
                    registry_name=_scenario_registry_name(imported=imported, module=module, cls=classes[0]),
                )
            )
    return sorted(contributions, key=lambda item: item.registry_name)


def _module_defined_subclasses(*, module: ModuleType, base_class: type[Scenario]) -> list[type[Scenario]]:
    return [
        candidate
        for _, candidate in inspect.getmembers(module, inspect.isclass)
        if candidate is not base_class
        and issubclass(candidate, base_class)
        and not inspect.isabstract(candidate)
        and candidate.__module__ == module.__name__
    ]


def _scenario_registry_name(
    *,
    imported: ImportedPlugin,
    module: ModuleType,
    cls: type[Scenario],
) -> str:
    prefix = next(
        (
            owned
            for owned in sorted(imported.prepared.owned_module_prefixes, key=len, reverse=True)
            if module.__name__ == owned or module.__name__.startswith(f"{owned}.")
        ),
        "",
    )
    relative = module.__name__[len(prefix) :].lstrip(".") if prefix else module.__name__
    for marker in ("scenario.scenarios.", "scenarios."):
        if marker in relative:
            relative = relative.split(marker, 1)[1]
            break
    if relative:
        return relative
    source = imported.prepared.spec.source
    if source is not None and source.is_file():
        return source.stem
    return class_name_to_snake_case(cls.__name__, suffix="Scenario")


def _name_owned_by_any(*, name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
