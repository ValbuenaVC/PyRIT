# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Activate private scenarios and attack techniques from configured plug-ins."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pyrit.datasets import SeedDatasetProvider
from pyrit.exceptions import (
    PluginCollisionError,
    PluginLoadError,
    PluginRegisteredNothingError,
    PluginValidationError,
)
from pyrit.registry import AttackTechniqueRegistry, ScenarioRegistry
from pyrit.setup.plugin_discovery import (
    ScenarioContribution,
    TechniqueContribution,
    discover_scenarios,
    discover_techniques,
    import_plugin_async,
)
from pyrit.setup.plugin_formats import PreparedPlugin, SourcePluginFormat, WheelPluginFormat
from pyrit.setup.plugin_spec import PluginFormat
from pyrit.setup.pyrit_initializer import PyRITInitializer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyrit.models import ComponentIdentifier
    from pyrit.registry import ScenarioMetadata
    from pyrit.registry.instance_registry import DefaultInstanceRegistry, RegistryEntry
    from pyrit.scenario import AttackTechniqueFactory, Scenario
    from pyrit.setup.plugin_spec import PluginSpec

logger = logging.getLogger(__name__)

_REMEDIATION = "Fix or remove the plug-in entry from .pyrit_conf, then restart PyRIT."


async def load_plugins_if_configured_async(*, plugins: Sequence[PluginSpec]) -> None:
    """
    Activate the configured V1 plug-in.

    Args:
        plugins: The normalized plug-in declarations.

    Raises:
        PluginLoadError: If activation fails.
        ValueError: If more than one plug-in is supplied.
    """
    if not plugins:
        return
    if len(plugins) > 1:
        raise ValueError("V1 supports one plug-in at a time; plug-in composition is not supported.")
    await PluginLoader(spec=plugins[0]).load_async()


class PluginInitializer(PyRITInitializer):
    """Privileged config-owned initializer that activates one scenario plug-in."""

    def __init__(self, *, plugins: Sequence[PluginSpec]) -> None:
        """
        Initialize the privileged plug-in loader.

        Args:
            plugins (Sequence[PluginSpec]): The normalized plug-in declarations.

        Raises:
            ValueError: If V1 receives anything other than one plug-in.
        """
        super().__init__()
        if len(plugins) != 1:
            raise ValueError("PluginInitializer requires exactly one plug-in in V1.")
        self._plugins = tuple(plugins)

    @property
    def plugins(self) -> list[PluginSpec]:
        """The normalized plug-in declarations."""
        return list(self._plugins)

    async def initialize_async(self) -> None:
        """Activate the configured plug-in before user-configured initializers."""
        await load_plugins_if_configured_async(plugins=self._plugins)


@dataclass
class _RegistrySnapshot:
    scenario_classes: dict[str, type[Scenario]]
    scenario_metadata: dict[str, ScenarioMetadata] | None
    scenario_discovered: bool
    technique_entries: dict[str, RegistryEntry[AttackTechniqueFactory]]
    technique_metadata: list[ComponentIdentifier] | None
    providers: dict[str, type[SeedDatasetProvider]]
    sys_path: list[str]
    module_names: set[str]

    @classmethod
    def capture(cls) -> _RegistrySnapshot:
        scenario_registry = ScenarioRegistry.get_registry_singleton()
        technique_registry = AttackTechniqueRegistry.get_registry_singleton()
        technique_instances = cast(
            "DefaultInstanceRegistry[AttackTechniqueFactory]",
            technique_registry.instances,
        )
        return cls(
            scenario_classes=dict(scenario_registry._classes),
            scenario_metadata=(
                dict(scenario_registry._metadata_cache) if scenario_registry._metadata_cache is not None else None
            ),
            scenario_discovered=scenario_registry._discovered,
            technique_entries=dict(technique_instances._registry_items),
            technique_metadata=(
                list(technique_instances._metadata_cache) if technique_instances._metadata_cache is not None else None
            ),
            providers=dict(SeedDatasetProvider._registry),
            sys_path=list(sys.path),
            module_names=set(sys.modules),
        )

    def restore(self, *, owned_prefixes: tuple[str, ...] = ()) -> None:
        scenario_registry = ScenarioRegistry.get_registry_singleton()
        scenario_registry._classes = dict(self.scenario_classes)
        scenario_registry._metadata_cache = dict(self.scenario_metadata) if self.scenario_metadata is not None else None
        scenario_registry._discovered = self.scenario_discovered

        technique_instances = cast(
            "DefaultInstanceRegistry[AttackTechniqueFactory]",
            AttackTechniqueRegistry.get_registry_singleton().instances,
        )
        technique_instances._registry_items = dict(self.technique_entries)
        technique_instances._metadata_cache = (
            list(self.technique_metadata) if self.technique_metadata is not None else None
        )

        SeedDatasetProvider._registry.clear()
        SeedDatasetProvider._registry.update(self.providers)
        sys.path[:] = self.sys_path
        for name in list(sys.modules):
            if name not in self.module_names and _name_owned_by_any(name=name, prefixes=owned_prefixes):
                del sys.modules[name]

    def restore_unsupported_provider_side_effects(self) -> None:
        """Remove provider registrations because datasets are outside the V1 contract."""
        SeedDatasetProvider._registry.clear()
        SeedDatasetProvider._registry.update(self.providers)


class PluginLoader:
    """Prepare, discover, validate, and transactionally register one plug-in."""

    def __init__(self, *, spec: PluginSpec) -> None:
        """
        Initialize the loader.

        Args:
            spec (PluginSpec): The normalized source or wheel declaration.
        """
        self._spec = spec

    async def load_async(self) -> None:
        """
        Activate the configured plug-in.

        Raises:
            PluginLoadError: If preparation, discovery, validation, or registration fails.
        """
        snapshot = _RegistrySnapshot.capture()
        prepared: PreparedPlugin | None = None
        try:
            prepared = await self._prepare_async()
            imported = await import_plugin_async(prepared=prepared)
            self._reject_import_time_registration(snapshot=snapshot)
            scenarios = discover_scenarios(imported=imported)
            techniques = discover_techniques(imported=imported)
            self._validate_names(scenarios=scenarios, techniques=techniques)
            self._register(scenarios=scenarios, techniques=techniques)
            snapshot.restore_unsupported_provider_side_effects()
            logger.info(
                "Loaded plug-in '%s': %d scenario(s), %d attack technique(s).",
                self._spec.name,
                len(scenarios),
                len(techniques),
            )
        except Exception as exc:
            prefixes = prepared.owned_module_prefixes if prepared else ()
            snapshot.restore(owned_prefixes=prefixes)
            message = f"Failed to load plug-in '{self._spec.name}': {exc} {_REMEDIATION}"
            if isinstance(exc, PluginLoadError):
                raise type(exc)(message) from exc
            raise PluginLoadError(message) from exc

    async def _prepare_async(self) -> PreparedPlugin:
        if self._spec.format is PluginFormat.SOURCE:
            return await SourcePluginFormat().prepare_async(spec=self._spec)
        if self._spec.format is PluginFormat.WHEEL:
            prepared = await WheelPluginFormat().prepare_async(spec=self._spec)
            self._warn_on_version_drift(prepared=prepared)
            return prepared
        raise PluginValidationError(f"Unsupported plug-in format: {self._spec.format}")

    @staticmethod
    def _reject_import_time_registration(*, snapshot: _RegistrySnapshot) -> None:
        scenario_registry = ScenarioRegistry.get_registry_singleton()
        technique_instances = cast(
            "DefaultInstanceRegistry[AttackTechniqueFactory]",
            AttackTechniqueRegistry.get_registry_singleton().instances,
        )
        technique_entries = technique_instances._registry_items
        if scenario_registry._classes != snapshot.scenario_classes or technique_entries != snapshot.technique_entries:
            raise PluginValidationError(
                "Plug-in source registered components during import. V1 plug-ins must expose definitions "
                "and let the framework register them transactionally."
            )

    @staticmethod
    def _validate_names(
        *,
        scenarios: Sequence[ScenarioContribution],
        techniques: Sequence[TechniqueContribution],
    ) -> None:
        if not scenarios and not techniques:
            raise PluginRegisteredNothingError("Plug-in contributed no scenarios or attack techniques.")

        scenario_registry = ScenarioRegistry.get_registry_singleton()
        builtin_scenarios = set(scenario_registry.get_class_names())
        collisions = sorted(item.registry_name for item in scenarios if item.registry_name in builtin_scenarios)
        if collisions:
            raise PluginCollisionError(f"Scenario name collision(s): {collisions}.")

        from pyrit.setup.initializers.techniques import build_technique_factories

        builtin_techniques = {factory.name for factory in build_technique_factories()}
        registered_techniques = set(AttackTechniqueRegistry.get_registry_singleton().get_factories())
        technique_collisions = sorted(
            item.factory.name for item in techniques if item.factory.name in builtin_techniques | registered_techniques
        )
        if technique_collisions:
            raise PluginCollisionError(f"Attack technique name collision(s): {technique_collisions}.")

    def _register(
        self,
        *,
        scenarios: Sequence[ScenarioContribution],
        techniques: Sequence[TechniqueContribution],
    ) -> None:
        from pyrit.setup.initializers.techniques import build_technique_factories

        technique_registry = AttackTechniqueRegistry.get_registry_singleton()
        technique_registry.register_from_factories(build_technique_factories(groups=["core"]))
        for contribution in techniques:
            technique_registry.register_contributed_factory(
                factory=contribution.factory,
                plugin_name=self._spec.name or "",
                scenario_names=contribution.scenario_names,
            )

        scenario_registry = ScenarioRegistry.get_registry_singleton()
        for contribution in scenarios:
            scenario_registry.register_contributed_scenario(
                scenario_class=contribution.scenario_class,
                name=contribution.registry_name,
            )
            try:
                scenario_registry.get_class_metadata(contribution.scenario_class)
            except Exception as exc:
                raise PluginValidationError(
                    f"Scenario '{contribution.registry_name}' could not build registry metadata: {exc}"
                ) from exc

    @staticmethod
    def _warn_on_version_drift(*, prepared: PreparedPlugin) -> None:
        declared = prepared.declared_pyrit_version
        if not declared:
            return
        import pyrit

        running = getattr(pyrit, "__version__", "") or ""
        if _major_minor(declared) == _major_minor(running) and _major_minor(declared) is not None:
            return
        logger.warning(
            "PLUGIN VERSION DRIFT: plug-in '%s' declares pyrit %s but pyrit %s is running. "
            "Compatibility is the artifact author's responsibility.",
            prepared.spec.name,
            declared,
            running or "(unknown)",
        )


def _major_minor(version: str) -> tuple[int, int] | None:
    parts = version.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _name_owned_by_any(*, name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
