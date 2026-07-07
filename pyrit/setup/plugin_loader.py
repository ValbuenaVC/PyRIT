# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Load a non-disclosable PyRIT plug-in from a pre-built wheel at initialization time.

A plug-in is a pure-Python wheel that ships dataset providers and/or scenarios that
must not live in the public PyRIT repo. The loader **extracts** the wheel (stdlib
``zipfile`` — never ``pip``/``.venv``) into ``.plugin/<name>/``, prepends that directory
to ``sys.path``, imports the package (so ``SeedDatasetProvider`` subclasses self-register),
and runs the plug-in's bootstrap (a top-level ``register()`` callable or a shipped
``PyRITInitializer`` subclass) which registers the plug-in's scenarios.

``load_plugin_if_configured_async`` is invoked as a guaranteed-first phase inside
``initialize_pyrit_async`` — after central memory is set and **before** any configured
initializers run — so plug-in datasets and scenarios are registered before
``LoadDefaultDatasets`` / ``PreloadScenarioMetadata`` read the registry. Ordering is
therefore true by construction, without relying on ``.pyrit_conf`` list position. It is a
no-op when ``PLUGIN_WHEEL`` is unset.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import pkgutil
import shutil
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from pyrit.setup.pyrit_initializer import PyRITInitializer

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

    from pyrit.registry import ScenarioRegistry

logger = logging.getLogger(__name__)

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})


async def load_plugin_if_configured_async(*, fail_open: bool | None = None) -> None:
    """
    Load the plug-in referenced by ``PLUGIN_WHEEL`` if one is configured.

    Convenience entry point invoked by ``initialize_pyrit_async``. A no-op when
    ``PLUGIN_WHEEL`` is unset.

    Args:
        fail_open: If provided, overrides the ``PLUGIN_FAIL_OPEN`` environment variable.
            When True, a plug-in that fails to load is skipped with a warning.

    Raises:
        PluginLoadError: If the plug-in is configured but fails to load and fail-open is
            not enabled.
    """
    await PluginLoader(fail_open=fail_open).load_async()


def _name_owned_by(module_name: str, package_name: str) -> bool:
    """
    Return whether a module name belongs to the given plug-in package.

    Args:
        module_name: A dotted module name.
        package_name: The plug-in's top-level package name.

    Returns:
        bool: True if ``module_name`` is the package or one of its submodules.
    """
    return module_name == package_name or module_name.startswith(f"{package_name}.")


def _module_owned_by(cls: type, package_name: str) -> bool:
    """
    Return whether ``cls`` is defined within the given plug-in package.

    Args:
        cls: The class to check.
        package_name: The plug-in's top-level package name.

    Returns:
        bool: True if the class's module is the package or one of its submodules.
    """
    return _name_owned_by(cls.__module__ or "", package_name)


class PluginLoadError(RuntimeError):
    """Raised when a configured plug-in fails to load and fail-open is not enabled."""


class PluginLoader:
    """
    Extract and register a PyRIT plug-in wheel referenced by ``PLUGIN_WHEEL``.

    No-op unless ``PLUGIN_WHEEL`` is set. When set, the wheel is extracted to
    ``.plugin/<name>/`` (never installed), imported, and its bootstrap is run so its
    datasets and scenarios register like built-ins. Fails closed by default; set
    ``fail_open`` (constructor / ``initialize_pyrit_async`` param) or ``PLUGIN_FAIL_OPEN``
    to continue without the plug-in when it cannot be loaded.
    """

    def __init__(self, *, fail_open: bool | None = None) -> None:
        """
        Initialize the loader.

        Args:
            fail_open: If provided, overrides ``PLUGIN_FAIL_OPEN``. When True, a plug-in
                that fails to load is skipped with a warning instead of raising.
        """
        self._explicit_fail_open = fail_open

    async def load_async(self) -> None:
        """
        Load the plug-in referenced by ``PLUGIN_WHEEL`` (no-op when unset).

        Raises:
            PluginLoadError: If the plug-in is configured but fails to load and fail-open
                is not enabled.
        """
        wheel_env = os.getenv("PLUGIN_WHEEL")
        if not wheel_env:
            logger.debug("PLUGIN_WHEEL is not set; plug-in loading is a no-op.")
            return

        fail_open = self._resolve_fail_open()

        try:
            await self._load_plugin_async(wheel_path=Path(wheel_env).expanduser())
        except Exception as exc:
            if fail_open:
                logger.warning(
                    "Plug-in from PLUGIN_WHEEL='%s' failed to load; fail_open is set so continuing without it: %s",
                    wheel_env,
                    exc,
                )
                return
            raise PluginLoadError(
                f"Failed to load plug-in from PLUGIN_WHEEL='{wheel_env}': {exc}. "
                "Remove the plug-in configuration, or enable fail-open (PLUGIN_FAIL_OPEN=true, or the "
                "initialize_pyrit_async(plugin_fail_open=True) parameter) to continue without it."
            ) from exc

    async def _load_plugin_async(self, *, wheel_path: Path) -> None:
        """
        Extract, import, bootstrap, and verify a single plug-in wheel.

        Global state (``sys.path``, imported plug-in modules, and the provider/scenario
        registries) is rolled back if the load fails, so a failed or fail-open load
        leaves no partial trace.

        Args:
            wheel_path: Path to the pre-built plug-in wheel on disk.

        Raises:
            FileNotFoundError: If ``wheel_path`` does not point to an existing file.
            ValueError: If the file is not a ``.whl`` or the plug-in registered nothing.
        """
        if not wheel_path.is_file():
            raise FileNotFoundError(f"PLUGIN_WHEEL does not point to an existing file: {wheel_path}")
        if wheel_path.suffix != ".whl":
            raise ValueError(f"PLUGIN_WHEEL must point to a .whl file, got: {wheel_path}")

        extract_dir = self._extract_wheel(wheel_path=wheel_path)
        package_name = self._resolve_package_name(extract_dir=extract_dir)

        from pyrit.datasets.seed_datasets.seed_dataset_provider import SeedDatasetProvider
        from pyrit.registry import ScenarioRegistry

        scenario_registry = ScenarioRegistry.get_registry_singleton()
        provider_snapshot = dict(SeedDatasetProvider._registry)
        scenario_snapshot = dict(scenario_registry._classes)
        modules_snapshot = {name for name in sys.modules if _name_owned_by(name, package_name)}
        syspath_entry = str(extract_dir)
        added_to_syspath = syspath_entry not in sys.path
        if added_to_syspath:
            sys.path.insert(0, syspath_entry)

        try:
            logger.info("Importing plug-in package '%s'", package_name)
            module = importlib.import_module(package_name)
            self._verify_module_location(module=module, extract_dir=extract_dir, package_name=package_name)
            self._import_submodules(module=module, package_name=package_name)

            await self._run_bootstrap_async(package_name=package_name, module=module)

            provider_count, scenario_count = self._count_registered(
                package_name=package_name, scenario_registry=scenario_registry
            )
            if not provider_count and not scenario_count:
                raise ValueError(
                    f"Plug-in package '{package_name}' imported successfully but registered no datasets or "
                    "scenarios. The wheel is likely mis-packaged (imports cleanly yet loads nothing)."
                )

            dataset_collisions = self._warn_on_dataset_name_collisions(package_name=package_name)
        except Exception:
            self._rollback(
                package_name=package_name,
                syspath_entry=syspath_entry if added_to_syspath else None,
                modules_snapshot=modules_snapshot,
                provider_snapshot=provider_snapshot,
                scenario_registry=scenario_registry,
                scenario_snapshot=scenario_snapshot,
            )
            raise

        logger.info(
            "Loaded plug-in '%s': %d dataset provider(s), %d scenario(s) registered; %d dataset name collision(s)%s.",
            package_name,
            provider_count,
            scenario_count,
            len(dataset_collisions),
            " (see PLUGIN DATASET SHADOWED warnings above)" if dataset_collisions else "",
        )

    def _extract_wheel(self, *, wheel_path: Path) -> Path:
        """
        Extract the wheel into ``.plugin/<wheel-stem>/``, reusing a cached extraction.

        Extraction is atomic: the wheel is unpacked into a temporary sibling directory and
        moved into place only on success, so a crash mid-extraction never leaves a partial
        tree that would later be treated as a valid cache.

        Args:
            wheel_path: Path to the plug-in wheel.

        Returns:
            Path: The directory the wheel was extracted to.
        """
        base_dir = self._plugin_base_dir()
        base_dir.mkdir(parents=True, exist_ok=True)

        extract_dir = base_dir / wheel_path.stem
        if extract_dir.is_dir() and any(extract_dir.iterdir()):
            logger.info("Reusing cached plug-in extraction at %s", extract_dir)
            return extract_dir

        tmp_dir = base_dir / f".{wheel_path.stem}.tmp-{os.getpid()}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        try:
            with zipfile.ZipFile(wheel_path) as wheel_zip:
                wheel_zip.extractall(tmp_dir)
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            os.replace(tmp_dir, extract_dir)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info("Extracted plug-in wheel '%s' to %s", wheel_path.name, extract_dir)
        return extract_dir

    @staticmethod
    def _plugin_base_dir() -> Path:
        """
        Resolve the base directory for plug-in extractions.

        Uses ``PLUGIN_DIR`` when set, otherwise ``<pyrit home>/.plugin``.

        Returns:
            Path: The resolved plug-in base directory.
        """
        override = os.getenv("PLUGIN_DIR")
        if override:
            return Path(override).expanduser().resolve()
        from pyrit.common import path

        return Path(path.HOME_PATH, ".plugin").resolve()

    @staticmethod
    def _resolve_package_name(*, extract_dir: Path) -> str:
        """
        Determine the plug-in's top-level import package.

        Resolution order: ``PLUGIN_PACKAGE`` env var, then ``*.dist-info/top_level.txt``,
        then the single importable top-level directory in the extraction.

        Args:
            extract_dir: The directory the wheel was extracted to.

        Returns:
            str: The top-level package name to import.

        Raises:
            ValueError: If the package cannot be unambiguously determined.
        """
        explicit = os.getenv("PLUGIN_PACKAGE")
        if explicit:
            return explicit

        for dist_info in sorted(extract_dir.glob("*.dist-info")):
            top_level = dist_info / "top_level.txt"
            if top_level.is_file():
                for line in top_level.read_text(encoding="utf-8").splitlines():
                    name = line.strip()
                    if name:
                        return name

        candidates = sorted(
            child.name
            for child in extract_dir.iterdir()
            if child.is_dir()
            and not child.name.endswith(".dist-info")
            and not child.name.endswith(".data")
            and (child / "__init__.py").is_file()
        )
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(
                f"Could not find an importable top-level package in {extract_dir}. "
                "Set PLUGIN_PACKAGE to the plug-in's package name."
            )
        raise ValueError(
            f"Found multiple top-level packages in {extract_dir}: {candidates}. Set PLUGIN_PACKAGE to disambiguate."
        )

    @staticmethod
    def _verify_module_location(*, module: ModuleType, extract_dir: Path, package_name: str) -> None:
        """
        Verify the imported package resolves inside the extraction directory.

        Guards against an installed package of the same name shadowing the extracted
        plug-in — a silent failure where import succeeds but the wheel's code/data is
        ignored.

        Args:
            module: The imported plug-in package module.
            extract_dir: The directory the wheel was extracted to.
            package_name: The plug-in's top-level package name.

        Raises:
            ValueError: If the imported package resolves outside ``extract_dir``.
        """
        extract_resolved = extract_dir.resolve()
        raw_locations = list(getattr(module, "__path__", []) or [])
        module_file = getattr(module, "__file__", None)
        if module_file:
            raw_locations.append(module_file)

        locations = [Path(location).resolve() for location in raw_locations if location]
        if not locations:
            return
        if not any(location.is_relative_to(extract_resolved) for location in locations):
            raise ValueError(
                f"Imported package '{package_name}' resolved to {locations[0]} which is outside the "
                f"plug-in extraction directory {extract_resolved}. An installed package with the same "
                "name is likely shadowing the plug-in; set PLUGIN_PACKAGE or resolve the name conflict."
            )

    @staticmethod
    def _import_submodules(*, module: ModuleType, package_name: str) -> None:
        """
        Import every submodule of the plug-in package.

        Ensures dataset providers self-register and bootstrap initializers become
        discoverable even when the package ``__init__`` does not import them. Import
        errors surface (plug-in dependencies must be pre-satisfied — fail loud).

        Args:
            module: The imported plug-in package module.
            package_name: The plug-in's top-level package name.
        """
        module_path = getattr(module, "__path__", None)
        if not module_path:
            return  # Single-module plug-in (not a package); nothing to walk.

        def _raise_on_error(name: str) -> None:
            raise ImportError(f"Failed to import plug-in submodule '{name}'")

        for submodule in pkgutil.walk_packages(module_path, prefix=f"{package_name}.", onerror=_raise_on_error):
            importlib.import_module(submodule.name)

    async def _run_bootstrap_async(self, *, package_name: str, module: ModuleType) -> None:
        """
        Run the plug-in's bootstrap so its scenarios register.

        Prefers a top-level ``register()`` callable on the package, then any
        ``PyRITInitializer`` subclass defined within the package. If neither exists the
        plug-in is assumed to register everything on import (datasets-only plug-ins).

        Args:
            package_name: The plug-in's top-level package name.
            module: The imported plug-in package module.
        """
        register = getattr(module, "register", None)
        if callable(register):
            logger.info("Running plug-in bootstrap register() from '%s'", package_name)
            result = register()
            if inspect.isawaitable(result):
                await result
            return

        initializer_classes = self._find_plugin_initializers(package_name=package_name)
        if initializer_classes:
            for initializer_class in initializer_classes:
                logger.info("Running plug-in bootstrap initializer %s", initializer_class.__name__)
                await initializer_class().initialize_async()
            return

        logger.info(
            "Plug-in '%s' exposes no register() or PyRITInitializer bootstrap; relying on "
            "import-time registration only.",
            package_name,
        )

    @staticmethod
    def _find_plugin_initializers(*, package_name: str) -> list[type[PyRITInitializer]]:
        """
        Find concrete ``PyRITInitializer`` subclasses defined within the plug-in package.

        Args:
            package_name: The plug-in's top-level package name.

        Returns:
            list[type[PyRITInitializer]]: Bootstrap initializer classes owned by the plug-in.
        """
        prefix = f"{package_name}."
        found: list[type[PyRITInitializer]] = []
        seen: set[type[PyRITInitializer]] = set()

        stack: list[type[PyRITInitializer]] = list(PyRITInitializer.__subclasses__())
        while stack:
            cls = stack.pop()
            if cls in seen:
                continue
            seen.add(cls)
            stack.extend(cls.__subclasses__())

            module_name = cls.__module__ or ""
            if inspect.isabstract(cls):
                continue
            if module_name == package_name or module_name.startswith(prefix):
                found.append(cls)
        return found

    @staticmethod
    def _count_registered(*, package_name: str, scenario_registry: ScenarioRegistry) -> tuple[int, int]:
        """
        Count providers and scenarios registered by the plug-in package.

        Both are counted by matching each registered class's module against the plug-in
        package, so the check is precise to this plug-in and safe to re-run.

        Args:
            package_name: The plug-in's top-level package name.
            scenario_registry: The scenario registry singleton the bootstrap registered into.

        Returns:
            tuple[int, int]: (dataset provider count, scenario count) owned by the plug-in.
        """
        from pyrit.datasets.seed_datasets.seed_dataset_provider import SeedDatasetProvider

        provider_count = sum(
            1 for cls in SeedDatasetProvider.get_all_providers().values() if _module_owned_by(cls, package_name)
        )

        # Read the raw class catalog directly: this snapshot must not trigger built-in
        # discovery, and the plug-in's register_class writes straight into it.
        scenario_count = sum(1 for cls in scenario_registry._classes.values() if _module_owned_by(cls, package_name))

        return provider_count, scenario_count

    @staticmethod
    def _warn_on_dataset_name_collisions(*, package_name: str) -> list[str]:
        """
        Warn loudly when a plug-in dataset name collides with an existing dataset name.

        The dataset resolver treats central memory as authoritative and only consults a
        provider when memory has no seeds for that ``dataset_name``. Once a same-named
        dataset is in memory, a scan uses it and never consults the plug-in's provider, so
        the plug-in's copy is silently bypassed. Any collision with **another** registered
        provider's ``dataset_name`` is surfaced prominently at load time so the mismatch is
        never silent.

        This compares the **provider registry**, not live memory, on purpose. At this phase
        memory is not populated yet, and a live-memory check would false-positive on the
        plug-in's own datasets persisted from a prior run (the seed rows carry no trustworthy
        source, so "already in memory" cannot be told apart from "this plug-in loaded it last
        run" — it is fundamentally undecidable at load time). The registry check is a
        **conservative proxy** for the shadowing that ``LoadDefaultDatasets`` will cause by
        loading provider datasets into memory: if the operator's config does not run
        ``load_default_datasets`` (or loads only a tag subset), a built-in name may not actually
        land in memory and this warning can fire without real shadowing. Over-warning is the
        safe direction — do NOT "fix" this into a memory check (it reintroduces the false
        positives). Governing principle: a guard's value is its precision — a check that cries
        wolf on legitimate plug-in data every run desensitizes operators and defeats itself for
        the real collision, so false-positive-free with a documented gap beats high-recall-but-
        noisy. Hard enforcement that can tell a real mismatch from a harmless name coincidence
        belongs to the scenario's declared required-dataset-names / expected-source mechanism
        (which knows the operator's intent), not this loader, and is intentionally not gated
        behind ``PLUGIN_FAIL_OPEN``.

        Args:
            package_name: The plug-in's top-level package name.

        Returns:
            list[str]: The sorted colliding dataset names (empty when there are none).
        """
        from pyrit.datasets.seed_datasets.seed_dataset_provider import SeedDatasetProvider

        def _safe_name(provider_class: type[SeedDatasetProvider]) -> str | None:
            try:
                return provider_class().dataset_name
            except Exception:
                return None

        providers = SeedDatasetProvider.get_all_providers()

        # Map dataset_name -> owning provider class name(s), split into the plug-in's own
        # providers vs. everything else. "Other" deliberately EXCLUDES the plug-in's own
        # providers, so a plug-in shipping multiple datasets (or a re-run) never self-flags.
        plugin_owned: dict[str, str] = {}
        other_owned: dict[str, list[str]] = {}
        for class_name, provider_class in providers.items():
            name = _safe_name(provider_class)
            if name is None:
                continue
            if _module_owned_by(provider_class, package_name):
                plugin_owned.setdefault(name, class_name)
            else:
                other_owned.setdefault(name, []).append(class_name)

        collisions = sorted(plugin_owned.keys() & other_owned.keys())
        for name in collisions:
            logger.warning(
                "PLUGIN DATASET SHADOWED: plug-in '%s' provider %s registers dataset_name '%s', which is "
                "already provided by %s. Central memory is authoritative, so a scan will use the existing "
                "dataset and the plug-in's copy will NOT take effect. Rename the plug-in dataset to a unique name.",
                package_name,
                plugin_owned[name],
                name,
                ", ".join(sorted(other_owned[name])),
            )
        return collisions

    @staticmethod
    def _rollback(
        *,
        package_name: str,
        syspath_entry: str | None,
        modules_snapshot: set[str],
        provider_snapshot: Mapping[str, type],
        scenario_registry: ScenarioRegistry,
        scenario_snapshot: Mapping[str, type],
    ) -> None:
        """
        Undo the partial global-state changes made while loading a plug-in.

        Removes the plug-in's ``sys.path`` entry, the modules it newly imported, and any
        provider/scenario registrations it added, so a failed (or fail-open) load leaves
        PyRIT as if the plug-in had never been loaded. State present before the load —
        including modules that already existed and built-ins discovered meanwhile — is
        preserved.

        Args:
            package_name: The plug-in's top-level package name.
            syspath_entry: The ``sys.path`` entry to remove, or None if it was already present.
            modules_snapshot: Package-owned module names present before the load.
            provider_snapshot: Provider registry contents captured before the load.
            scenario_registry: The scenario registry singleton to clean up.
            scenario_snapshot: Scenario catalog contents captured before the load.
        """
        from pyrit.datasets.seed_datasets.seed_dataset_provider import SeedDatasetProvider

        if syspath_entry and syspath_entry in sys.path:
            sys.path.remove(syspath_entry)

        for name in [m for m in sys.modules if _name_owned_by(m, package_name) and m not in modules_snapshot]:
            del sys.modules[name]

        for key, cls in list(SeedDatasetProvider._registry.items()):
            if key not in provider_snapshot and _module_owned_by(cls, package_name):
                del SeedDatasetProvider._registry[key]

        removed_scenario = False
        for name, cls in list(scenario_registry._classes.items()):
            if name not in scenario_snapshot and _module_owned_by(cls, package_name):
                del scenario_registry._classes[name]
                removed_scenario = True
        if removed_scenario:
            scenario_registry._metadata_cache = None

    def _resolve_fail_open(self) -> bool:
        """
        Resolve the fail-open setting from the explicit value or the environment.

        Precedence: the explicit ``fail_open`` passed to the constructor (e.g. from
        ``initialize_pyrit_async``), then the ``PLUGIN_FAIL_OPEN`` environment variable,
        otherwise fail-closed.

        Returns:
            bool: True if a failed plug-in load should be skipped with a warning.
        """
        if self._explicit_fail_open is not None:
            return self._explicit_fail_open

        env_value = os.getenv("PLUGIN_FAIL_OPEN")
        if env_value is not None:
            return self._coerce_bool(env_value)

        return False

    @staticmethod
    def _coerce_bool(value: str) -> bool:
        """
        Interpret a string as a boolean flag.

        Args:
            value: The raw string value.

        Returns:
            bool: True for common truthy tokens (1/true/yes/on, case-insensitive).
        """
        return str(value).strip().lower() in _TRUE_TOKENS
