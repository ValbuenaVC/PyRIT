# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Unit tests for the PyRIT plug-in loader.

These tests build a **mock plug-in wheel** at test time (no dependency on any real
plug-in) and exercise the full consumer mechanism: extract -> sys.path ->
import -> bootstrap -> assert-loaded, plus the accept-load-failures/fail-closed policy and the
silent-failure guards called out in the design brief.
"""

import inspect
import logging
import os
import sys
import textwrap
import uuid
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.datasets.seed_datasets.seed_dataset_provider import SeedDatasetProvider
from pyrit.memory import CentralMemory
from pyrit.models import SeedDataset
from pyrit.registry import ScenarioRegistry
from pyrit.setup import PluginSpec
from pyrit.setup.initialization import IN_MEMORY, initialize_pyrit_async
from pyrit.setup.plugin_loader import (
    PluginImportError,
    PluginInitializer,
    PluginLoadError,
    PluginRegisteredNothingError,
    PluginWheelNotFoundError,
    load_plugins_if_configured_async,
)

# ---------------------------------------------------------------------------
# Mock-wheel builder
# ---------------------------------------------------------------------------


class MockWheel:
    """Handle describing a built mock plug-in wheel."""

    def __init__(self, *, path: Path, package: str, scenario_name: str, dataset_name: str) -> None:
        self.path = path
        self.package = package
        self.scenario_name = scenario_name
        self.dataset_name = dataset_name


def _unique_package_name() -> str:
    """Return a unique, import-safe mock package name."""
    return f"mock_plugin_{uuid.uuid4().hex[:8]}"


def build_mock_wheel(
    dest_dir: Path,
    *,
    bootstrap: str = "initializer",
    include_provider: bool = True,
    include_scenario: bool = True,
    wire_init: bool = True,
    package_name: str | None = None,
) -> MockWheel:
    """
    Build a mock plug-in wheel in ``dest_dir`` and return a handle to it.

    Args:
        dest_dir: Directory to write the wheel source tree and .whl into.
        bootstrap: Bootstrap style: "initializer" (a PyRITInitializer subclass),
            "register" (a top-level register() callable), or "none".
        include_provider: Whether to ship a self-registering SeedDatasetProvider.
        include_scenario: Whether to ship a Scenario subclass.
        wire_init: Whether __init__.py imports the submodules. When False, submodules are
            shipped but not imported by __init__, exercising the loader's submodule walk.
        package_name: Optional explicit package name; a unique one is generated otherwise.

    Returns:
        MockWheel: The built wheel handle (path + package/scenario/dataset names).
    """
    package_name = package_name or _unique_package_name()
    scenario_name = f"airt.{package_name}"
    dataset_name = f"{package_name}_dataset"

    src = dest_dir / f"{package_name}_src"
    pkg = src / package_name
    pkg.mkdir(parents=True, exist_ok=True)

    imports = []
    if include_provider:
        imports.append("provider")
    if include_scenario:
        imports.append("scenario")
    if bootstrap in ("initializer", "initializer_raises", "register"):
        imports.append("bootstrap")

    init_lines = []
    if wire_init and imports:
        init_lines.append(f"from . import {', '.join(imports)}  # noqa: F401")
    if wire_init and bootstrap == "register":
        init_lines.append("from .bootstrap import register  # noqa: F401")
    (pkg / "__init__.py").write_text(("\n".join(init_lines) + "\n") if init_lines else "", encoding="utf-8")

    # __file__-relative dataset path (as a real plug-in ships). Only resolves on real disk.
    (pkg / "paths.py").write_text(
        textwrap.dedent(
            """\
            from pathlib import Path

            MOCK_ROOT = Path(__file__, "..").resolve()
            MOCK_DATASETS_PATH = Path(MOCK_ROOT, "datasets").resolve()
            """
        ),
        encoding="utf-8",
    )

    if include_provider:
        datasets = pkg / "datasets"
        datasets.mkdir(parents=True, exist_ok=True)
        (pkg / "provider.py").write_text(
            textwrap.dedent(
                f"""\
                from pyrit.datasets.seed_datasets.seed_dataset_provider import SeedDatasetProvider
                from pyrit.models.seeds.seed_dataset import SeedDataset

                from .paths import MOCK_DATASETS_PATH


                class MockProvider(SeedDatasetProvider):
                    @property
                    def dataset_name(self) -> str:
                        return "{dataset_name}"

                    async def fetch_dataset_async(self, *, cache: bool = True) -> SeedDataset:
                        return SeedDataset.from_yaml_file(MOCK_DATASETS_PATH / "seed.yaml")
                """
            ),
            encoding="utf-8",
        )
        (datasets / "seed.yaml").write_text(
            textwrap.dedent(
                f"""\
                dataset_name: {dataset_name}
                harm_categories:
                  - mock
                data_type: text
                description: mock dataset for plugin test
                authors:
                  - tester
                groups:
                  - test
                seeds:
                  - value: mock prompt one
                  - value: mock prompt two
                  - value: mock prompt three
                """
            ),
            encoding="utf-8",
        )

    if include_scenario:
        (pkg / "scenario.py").write_text(
            textwrap.dedent(
                """\
                from pyrit.scenario.scenarios.airt.rapid_response import RapidResponse


                class MockScenario(RapidResponse):
                    \"\"\"Mock plugin scenario for registration test.\"\"\"
                """
            ),
            encoding="utf-8",
        )

    if bootstrap == "initializer":
        (pkg / "bootstrap.py").write_text(
            textwrap.dedent(
                f"""\
                from pyrit.registry import ScenarioRegistry
                from pyrit.setup.pyrit_initializer import PyRITInitializer

                from .scenario import MockScenario


                class MockBootstrapInitializer(PyRITInitializer):
                    \"\"\"Register the mock plugin scenario.\"\"\"

                    async def initialize_async(self) -> None:
                        ScenarioRegistry.get_registry_singleton().register_class(
                            MockScenario, name="{scenario_name}"
                        )
                """
            ),
            encoding="utf-8",
        )
    elif bootstrap == "initializer_raises":
        (pkg / "bootstrap.py").write_text(
            textwrap.dedent(
                f"""\
                from pyrit.registry import ScenarioRegistry
                from pyrit.setup.pyrit_initializer import PyRITInitializer

                from .scenario import MockScenario


                class MockBootstrapInitializer(PyRITInitializer):
                    \"\"\"Register the scenario, then fail to test rollback.\"\"\"

                    async def initialize_async(self) -> None:
                        ScenarioRegistry.get_registry_singleton().register_class(
                            MockScenario, name="{scenario_name}"
                        )
                        raise RuntimeError("bootstrap failed after registering")
                """
            ),
            encoding="utf-8",
        )
    elif bootstrap == "register":
        (pkg / "bootstrap.py").write_text(
            textwrap.dedent(
                f"""\
                from pyrit.registry import ScenarioRegistry

                from .scenario import MockScenario


                def register() -> None:
                    ScenarioRegistry.get_registry_singleton().register_class(
                        MockScenario, name="{scenario_name}"
                    )
                """
            ),
            encoding="utf-8",
        )

    # Minimal dist-info without top_level.txt so package name inference is exercised.
    distinfo = src / f"{package_name}-0.0.1.dist-info"
    distinfo.mkdir(parents=True, exist_ok=True)
    (distinfo / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {package_name}\nVersion: 0.0.1\n", encoding="utf-8"
    )
    (distinfo / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n", encoding="utf-8"
    )
    (distinfo / "RECORD").write_text("", encoding="utf-8")

    wheel = dest_dir / f"{package_name}-0.0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(src):
            for file_name in files:
                file_path = Path(root) / file_name
                archive.write(file_path, str(file_path.relative_to(src)))

    return MockWheel(path=wheel, package=package_name, scenario_name=scenario_name, dataset_name=dataset_name)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def plugin_sandbox() -> Iterator[None]:
    """Snapshot and restore global import + registry state around each test."""
    sys_path_snapshot = list(sys.path)
    provider_snapshot = dict(SeedDatasetProvider._registry)

    yield

    sys.path[:] = sys_path_snapshot
    # Only drop the mock plug-in modules this suite imports; leave real pyrit modules
    # in place so re-imports don't create duplicate class objects for other tests.
    for name in [m for m in sys.modules if m == "mock_plugin" or m.startswith("mock_plugin_")]:
        del sys.modules[name]
    SeedDatasetProvider._registry.clear()
    SeedDatasetProvider._registry.update(provider_snapshot)
    ScenarioRegistry.reset_registry_singleton()


@contextmanager
def plugin_env(**overrides: str) -> Iterator[None]:
    """Patch os.environ so only the given PLUGIN_* overrides are present."""
    with patch.dict(os.environ, overrides, clear=False):
        for key in ("PLUGIN_DIR",):
            if key not in overrides:
                os.environ.pop(key, None)
        yield


def _spec(wheel: MockWheel, *, package: str | None = None) -> PluginSpec:
    """Build a PluginSpec from a mock wheel."""
    return PluginSpec(wheel=wheel.path, package=package)


async def load_plugin(
    wheel: MockWheel,
    plugin_dir: Path,
    *,
    package: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Run the plug-in loader against a single mock wheel with an isolated env."""
    with plugin_env(PLUGIN_DIR=str(plugin_dir), **(extra_env or {})):
        await load_plugins_if_configured_async(plugins=[_spec(wheel, package=package)])


# ---------------------------------------------------------------------------
# Privileged PluginInitializer
# ---------------------------------------------------------------------------


def test_core_initialize_has_no_plugin_parameters() -> None:
    parameters = inspect.signature(initialize_pyrit_async).parameters
    assert "plugins" not in parameters
    assert "plugin_accept_load_failures" not in parameters


async def test_plugin_initializer_delegates_to_loader() -> None:
    load_plugins_mock = AsyncMock()
    spec = PluginSpec(wheel=Path("plugin.whl"))
    initializer = PluginInitializer(plugins=[spec])

    with patch("pyrit.setup.plugin_loader.load_plugins_if_configured_async", load_plugins_mock):
        await initializer.initialize_async()

    load_plugins_mock.assert_awaited_once()
    assert list(load_plugins_mock.await_args.kwargs["plugins"]) == [spec]


async def test_plugin_initializer_runs_after_memory_setup() -> None:
    memory = MagicMock()
    initializer = PluginInitializer(plugins=[PluginSpec(wheel=Path("plugin.whl"))])

    async def _assert_memory_ready() -> None:
        assert CentralMemory.get_memory_instance() is memory

    with (
        patch("pyrit.setup.initialization.SQLiteMemory", return_value=memory),
        patch.object(initializer, "initialize_async", side_effect=_assert_memory_ready) as initialize_mock,
    ):
        await initialize_pyrit_async(IN_MEMORY, initializers=[initializer], env_files=[], silent=True)

    initialize_mock.assert_awaited_once()


def test_plugin_initializer_is_not_user_discoverable() -> None:
    from pyrit.registry import InitializerRegistry

    InitializerRegistry.reset_registry_singleton()
    registry = InitializerRegistry.get_registry_singleton()

    assert "plugin" not in registry.get_class_names()


# ---------------------------------------------------------------------------
# No-op behavior
# ---------------------------------------------------------------------------


async def test_no_op_when_no_plugins() -> None:
    """With an empty plug-in list the loader does nothing and registers nothing."""
    providers_before = dict(SeedDatasetProvider.get_all_providers())
    path_before = list(sys.path)

    with plugin_env():
        await load_plugins_if_configured_async(plugins=[])

    assert SeedDatasetProvider.get_all_providers() == providers_before
    assert sys.path == path_before


# ---------------------------------------------------------------------------
# Silent-failure trap: extraction, not zipimport
# ---------------------------------------------------------------------------


def test_raw_wheel_on_syspath_loses_datasets(tmp_path: Path) -> None:
    """A raw .whl on sys.path imports but __file__-relative datasets vanish (regression guard)."""
    wheel = build_mock_wheel(tmp_path, bootstrap="none", include_scenario=False)

    sys.path.insert(0, str(wheel.path))
    module = __import__(wheel.package)
    paths_module = __import__(f"{wheel.package}.paths", fromlist=["MOCK_DATASETS_PATH"])

    assert ".whl" in (module.__file__ or "")
    assert not paths_module.MOCK_DATASETS_PATH.exists()
    assert list(paths_module.MOCK_DATASETS_PATH.glob("**/*.yaml")) == []


def test_extracted_wheel_loads_datasets(tmp_path: Path) -> None:
    """Extracting the wheel to disk makes __file__-relative datasets resolve and load."""
    wheel = build_mock_wheel(tmp_path, bootstrap="none", include_scenario=False)
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    with zipfile.ZipFile(wheel.path) as archive:
        archive.extractall(extract_dir)

    sys.path.insert(0, str(extract_dir))
    paths_module = __import__(f"{wheel.package}.paths", fromlist=["MOCK_DATASETS_PATH"])

    yamls = list(paths_module.MOCK_DATASETS_PATH.glob("**/*.yaml"))
    assert len(yamls) == 1

    dataset = SeedDataset.from_yaml_file(yamls[0])
    assert len(dataset.seeds) == 3


# ---------------------------------------------------------------------------
# Loading via the initializer
# ---------------------------------------------------------------------------


async def test_load_registers_provider_on_import(tmp_path: Path) -> None:
    """Importing the plug-in package self-registers its SeedDatasetProvider."""
    wheel = build_mock_wheel(tmp_path)

    await load_plugin(wheel, tmp_path / ".plugin")

    assert "MockProvider" in SeedDatasetProvider.get_all_providers()


async def test_load_extracts_to_plugin_dir(tmp_path: Path) -> None:
    """The wheel is extracted (not installed) under the configured plug-in dir."""
    wheel = build_mock_wheel(tmp_path)
    plugin_dir = tmp_path / ".plugin"

    await load_plugin(wheel, plugin_dir)

    extract_dir = plugin_dir / wheel.path.stem
    assert (extract_dir / wheel.package / "__init__.py").is_file()
    assert (extract_dir / wheel.package / "datasets" / "seed.yaml").is_file()


async def test_scenario_registration_survives_discovery(tmp_path: Path) -> None:
    """A plug-in scenario registered before discovery coexists with built-ins afterwards."""
    wheel = build_mock_wheel(tmp_path, bootstrap="initializer")

    await load_plugin(wheel, tmp_path / ".plugin")

    registry = ScenarioRegistry.get_registry_singleton()
    assert registry._discovered is False  # register_class must not trigger discovery

    names = registry.get_class_names()  # triggers built-in discovery
    assert wheel.scenario_name in names
    assert "airt.rapid_response" in names

    mock_scenario = sys.modules[f"{wheel.package}.scenario"].MockScenario
    assert registry.get_class(wheel.scenario_name) is mock_scenario


async def test_register_callable_bootstrap(tmp_path: Path) -> None:
    """A plug-in exposing a top-level register() callable is bootstrapped too."""
    wheel = build_mock_wheel(tmp_path, bootstrap="register")

    await load_plugin(wheel, tmp_path / ".plugin")

    names = ScenarioRegistry.get_registry_singleton().get_class_names()
    assert wheel.scenario_name in names


async def test_ordering_scenario_visible_to_preload(tmp_path: Path) -> None:
    """The plug-in scenario is registered before a later PreloadScenarioMetadata read."""
    wheel = build_mock_wheel(tmp_path, bootstrap="initializer")

    await load_plugin(wheel, tmp_path / ".plugin")

    # get_class_names() is exactly what PreloadScenarioMetadata iterates; the plug-in
    # scenario being present proves it registered before that read would happen.
    names = ScenarioRegistry.get_registry_singleton().get_class_names()
    assert wheel.scenario_name in names


async def test_plugin_scenario_auto_registered_without_bootstrap(tmp_path: Path) -> None:
    """A plug-in's Scenario subclass is auto-registered by discovery even with no bootstrap."""
    wheel = build_mock_wheel(tmp_path, bootstrap="none", include_provider=False)

    await load_plugin(wheel, tmp_path / ".plugin")

    # The scenario is picked up purely by type-scoped discovery of the plug-in package,
    # so a scenario-only plug-in with no register()/initializer still loads.
    registry = ScenarioRegistry.get_registry_singleton()
    mock_scenario = sys.modules[f"{wheel.package}.scenario"].MockScenario
    assert mock_scenario in registry._classes.values()
    assert registry._discovered is False  # auto-registration must not trigger built-in discovery


async def test_bootstrap_registration_not_duplicated_by_auto_register(tmp_path: Path) -> None:
    """A scenario the bootstrap registers is not also re-registered under a fallback name."""
    wheel = build_mock_wheel(tmp_path, bootstrap="register")

    await load_plugin(wheel, tmp_path / ".plugin")

    registry = ScenarioRegistry.get_registry_singleton()
    mock_scenario = sys.modules[f"{wheel.package}.scenario"].MockScenario
    registered_names = [name for name, cls in registry._classes.items() if cls is mock_scenario]
    assert registered_names == [wheel.scenario_name]


async def test_datasets_only_plugin_loads_without_bootstrap(tmp_path: Path) -> None:
    """A datasets-only plug-in (no bootstrap, no scenario) loads via import-time registration."""
    wheel = build_mock_wheel(tmp_path, bootstrap="none", include_scenario=False)

    await load_plugin(wheel, tmp_path / ".plugin")

    assert "MockProvider" in SeedDatasetProvider.get_all_providers()


async def test_submodule_walk_discovers_unwired_components(tmp_path: Path) -> None:
    """Provider + bootstrap register even when __init__.py does not import them."""
    wheel = build_mock_wheel(tmp_path, bootstrap="initializer", wire_init=False)

    await load_plugin(wheel, tmp_path / ".plugin")

    assert "MockProvider" in SeedDatasetProvider.get_all_providers()
    assert wheel.scenario_name in ScenarioRegistry.get_registry_singleton().get_class_names()


async def test_shadowing_installed_package_is_rejected(tmp_path: Path) -> None:
    """An installed package of the same name shadowing the plug-in fails loudly."""
    wheel = build_mock_wheel(tmp_path)

    # The spec's package points at a stdlib package that imports from outside the extraction dir.
    with pytest.raises(PluginImportError, match="shadowing"):
        await load_plugin(wheel, tmp_path / ".plugin", package="json")


# ---------------------------------------------------------------------------
# Dataset name collision (memory-authoritative resolver guard)
# ---------------------------------------------------------------------------


async def test_colliding_dataset_name_warns_loudly(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A plug-in dataset_name that collides with an existing provider's name warns at load."""
    wheel = build_mock_wheel(tmp_path)
    colliding_name = wheel.dataset_name

    class CollidingProvider(SeedDatasetProvider):
        """Non-plug-in provider that already claims the plug-in's dataset name."""

        @property
        def dataset_name(self) -> str:
            return colliding_name

        async def fetch_dataset_async(self, *, cache: bool = True) -> SeedDataset:
            raise NotImplementedError

    with caplog.at_level(logging.WARNING, logger="pyrit.setup.plugin_loader"):
        await load_plugin(wheel, tmp_path / ".plugin")

    messages = [record.getMessage() for record in caplog.records]
    # Un-missable: greppable prefix, names the colliding dataset and BOTH providers.
    assert any(
        "PLUGIN DATASET SHADOWED:" in message
        and colliding_name in message
        and "MockProvider" in message
        and "CollidingProvider" in message
        for message in messages
    )


async def test_unique_dataset_name_does_not_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A plug-in whose dataset name is unique produces no collision warning."""
    wheel = build_mock_wheel(tmp_path)

    with caplog.at_level(logging.WARNING, logger="pyrit.setup.plugin_loader"):
        await load_plugin(wheel, tmp_path / ".plugin")

    assert not any("PLUGIN DATASET SHADOWED:" in record.getMessage() for record in caplog.records)


async def test_reload_does_not_self_flag(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Loading the same plug-in twice must not flag its own provider as a collision."""
    wheel = build_mock_wheel(tmp_path)
    plugin_dir = tmp_path / ".plugin"

    await load_plugin(wheel, plugin_dir)
    with caplog.at_level(logging.WARNING, logger="pyrit.setup.plugin_loader"):
        await load_plugin(wheel, plugin_dir)

    assert not any("PLUGIN DATASET SHADOWED:" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# Rollback on failure
# ---------------------------------------------------------------------------


async def test_failed_load_rolls_back_syspath(tmp_path: Path) -> None:
    """A failed load removes its own sys.path entry (fail-closed leaves no trace)."""
    wheel = build_mock_wheel(tmp_path, bootstrap="none", include_provider=False, include_scenario=False)
    plugin_dir = tmp_path / ".plugin"

    with pytest.raises(PluginLoadError):
        await load_plugin(wheel, plugin_dir)

    extract_dir = str(plugin_dir / wheel.path.stem)
    assert extract_dir not in sys.path


async def test_failing_bootstrap_rolls_back_partial_registration(tmp_path: Path) -> None:
    """A bootstrap that registers then raises has its registration rolled back."""
    wheel = build_mock_wheel(tmp_path, bootstrap="initializer_raises")
    plugin_dir = tmp_path / ".plugin"

    with pytest.raises(PluginLoadError):
        await load_plugin(wheel, plugin_dir)

    registry = ScenarioRegistry.get_registry_singleton()
    assert wheel.scenario_name not in registry._classes
    assert "MockProvider" not in SeedDatasetProvider.get_all_providers()
    assert str(plugin_dir / wheel.path.stem) not in sys.path


async def test_rollback_restores_overwritten_provider(tmp_path: Path) -> None:
    """A failed load restores a provider entry the plug-in overwrote (name collision)."""
    wheel = build_mock_wheel(tmp_path, bootstrap="initializer_raises")

    # SeedDatasetProvider keys by class name and the mock provider is "MockProvider";
    # occupy that key so the plug-in's import overwrites it.
    class _PreexistingProvider(SeedDatasetProvider):
        should_register = False

        @property
        def dataset_name(self) -> str:
            return "preexisting"

        async def fetch_dataset_async(self, *, cache: bool = True) -> SeedDataset:
            raise NotImplementedError

    SeedDatasetProvider._registry["MockProvider"] = _PreexistingProvider

    with pytest.raises(PluginLoadError):
        await load_plugin(wheel, tmp_path / ".plugin")

    # The original provider is restored, not deleted or left replaced by the plug-in's.
    assert SeedDatasetProvider._registry["MockProvider"] is _PreexistingProvider


async def test_rollback_restores_overwritten_scenario(tmp_path: Path) -> None:
    """A failed load restores a scenario entry the plug-in overwrote (name collision)."""
    from pyrit.scenario.scenarios.airt.rapid_response import RapidResponse

    wheel = build_mock_wheel(tmp_path, bootstrap="initializer_raises")

    class _PreexistingScenario(RapidResponse):
        """Sentinel scenario occupying the plug-in's registry name."""

    registry = ScenarioRegistry.get_registry_singleton()
    registry.register_class(_PreexistingScenario, name=wheel.scenario_name)

    with pytest.raises(PluginLoadError):
        await load_plugin(wheel, tmp_path / ".plugin")

    # The original scenario is restored, not deleted or left replaced by the plug-in's.
    assert registry._classes[wheel.scenario_name] is _PreexistingScenario


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


async def test_missing_wheel_fails_closed() -> None:
    """A configured-but-missing wheel raises PluginWheelNotFoundError by default (fail-closed)."""
    with plugin_env():
        with pytest.raises(PluginWheelNotFoundError, match="Failed to load plug-in"):
            await load_plugins_if_configured_async(plugins=[PluginSpec(wheel=Path("does_not_exist.whl"))])


async def test_empty_wheel_is_loud(tmp_path: Path) -> None:
    """A wheel that imports cleanly but registers nothing fails loudly."""
    wheel = build_mock_wheel(tmp_path, bootstrap="none", include_provider=False, include_scenario=False)

    with pytest.raises(PluginRegisteredNothingError, match="registered no datasets or scenarios"):
        await load_plugin(wheel, tmp_path / ".plugin")


async def test_non_whl_path_fails_closed(tmp_path: Path) -> None:
    """A wheel path that is not a .whl file fails closed with PluginWheelNotFoundError."""
    not_a_wheel = tmp_path / "plugin.zip"
    not_a_wheel.write_text("not a wheel", encoding="utf-8")

    with plugin_env():
        with pytest.raises(PluginWheelNotFoundError, match="Failed to load plug-in"):
            await load_plugins_if_configured_async(plugins=[PluginSpec(wheel=not_a_wheel)])


def test_error_subclasses_are_plugin_load_errors() -> None:
    """All specific plug-in errors subclass PluginLoadError so one except still catches them."""
    for error_cls in (PluginWheelNotFoundError, PluginImportError, PluginRegisteredNothingError):
        assert issubclass(error_cls, PluginLoadError)


async def test_wheel_with_path_traversal_member_fails_closed(tmp_path: Path) -> None:
    """A wheel containing a path-traversal member is rejected during safe extraction."""
    malicious = tmp_path / "evil-0.0.1-py3-none-any.whl"
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("evil_pkg/__init__.py", "")
        archive.writestr("../escape.py", "compromised = True")

    with plugin_env(PLUGIN_DIR=str(tmp_path / ".plugin")):
        with pytest.raises(PluginLoadError, match="Failed to load plug-in"):
            await load_plugins_if_configured_async(plugins=[PluginSpec(wheel=malicious)])

    # The traversal target was not written outside the extraction directory.
    assert not (tmp_path / "escape.py").exists()


# ---------------------------------------------------------------------------
# No-arg-instantiable contract
# ---------------------------------------------------------------------------


def test_non_no_arg_scenario_fails_metadata_cleanly() -> None:
    """A registered scenario that is not no-arg instantiable fails metadata build clearly."""
    from pyrit.scenario.scenarios.airt.rapid_response import RapidResponse

    class BadScenario(RapidResponse):
        """Scenario that violates the no-arg-instantiable contract."""

        def __init__(self, *, required_value: str) -> None:
            super().__init__()
            self._required_value = required_value

    registry = ScenarioRegistry()
    registry.register_class(BadScenario, name="airt.bad")  # signature-only validation passes

    with pytest.raises(TypeError, match="no arguments"):
        registry._build_metadata("airt.bad", BadScenario)


# ---------------------------------------------------------------------------
# Multiple plug-ins
# ---------------------------------------------------------------------------


async def test_multiple_plugins_rejected(tmp_path: Path) -> None:
    """V1 rejects plug-in composition."""
    wheel_a = build_mock_wheel(tmp_path / "a", package_name="mock_plugin_alpha")
    wheel_b = build_mock_wheel(tmp_path / "b", package_name="mock_plugin_beta")

    with plugin_env(PLUGIN_DIR=str(tmp_path / ".plugin")):
        with pytest.raises(ValueError, match="one plug-in"):
            await load_plugins_if_configured_async(plugins=[_spec(wheel_a), _spec(wheel_b)])
