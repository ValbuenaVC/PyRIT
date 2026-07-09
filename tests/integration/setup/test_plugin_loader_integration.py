# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Integration test: plug-in scenarios are discovered by ``ScenarioRegistry`` after init.

This exercises the full public loading path end-to-end: point ``PLUGIN_WHEEL`` at a
built wheel, run ``initialize_pyrit_async``, and assert every scenario the wheel ships
is registered in ``ScenarioRegistry``.

Two cases:

* A self-contained case builds a small wheel at test time and verifies its scenarios
  are discovered. This always runs and guards the mechanism in public CI.
* An injected case loads a wheel supplied out-of-band via environment variables and
  verifies all of its scenarios are discovered. It is skipped unless those variables
  are set, so the committed test depends on no specific external package. Point it at a
  real scenario wheel (for example in a downstream/private CI job) to guarantee that
  every scenario that wheel ships is picked up after initialization::

      PLUGIN_TEST_WHEEL=/path/to/plugin.whl
      PLUGIN_TEST_PACKAGE=the_plugin_package          # optional; enables package enumeration
      PLUGIN_TEST_SCENARIO_DIRS=/path/to/scenarios    # optional; os.pathsep-separated
"""

import inspect
import os
import sys
import textwrap
import uuid
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from pyrit.datasets.seed_datasets.seed_dataset_provider import SeedDatasetProvider
from pyrit.registry import ScenarioRegistry
from pyrit.registry.discovery import discover_in_directory
from pyrit.scenario.core import Scenario
from pyrit.setup import IN_MEMORY, PluginSpec, initialize_pyrit_async

# (module stem, class name, registry name) for the scenarios the self-contained wheel ships.
_MOCK_SCENARIOS = [
    ("alpha", "MockAlphaScenario", "airt.mock_alpha"),
    ("beta", "MockBetaScenario", "airt.mock_beta"),
    ("gamma", "MockGammaScenario", "airt.mock_gamma"),
]


# REV Should this be a Pytest fixture?
def _build_scenario_plugin_wheel(dest_dir: Path, *, package: str) -> Path:
    """
    Build a wheel whose package ships several scenarios and a ``register()`` bootstrap.

    Args:
        dest_dir: Directory to write the source tree and .whl into.
        package: Top-level package name for the wheel.

    Returns:
        Path: The built wheel.
    """
    src = dest_dir / f"{package}_src"
    pkg = src / package
    scenarios_pkg = pkg / "scenarios"
    scenarios_pkg.mkdir(parents=True, exist_ok=True)

    (pkg / "__init__.py").write_text("from .bootstrap import register  # noqa: F401\n", encoding="utf-8")
    (scenarios_pkg / "__init__.py").write_text("", encoding="utf-8")

    for stem, class_name, _registry_name in _MOCK_SCENARIOS:
        (scenarios_pkg / f"{stem}.py").write_text(
            textwrap.dedent(
                f"""\
                from pyrit.scenario.scenarios.airt.rapid_response import RapidResponse


                class {class_name}(RapidResponse):
                    \"\"\"Mock plug-in scenario {stem}.\"\"\"
                """
            ),
            encoding="utf-8",
        )

    imports = "\n".join(f"from .scenarios.{stem} import {class_name}" for stem, class_name, _ in _MOCK_SCENARIOS)
    registrations = "\n".join(
        f'    registry.register_class({class_name}, name="{registry_name}")'
        for _stem, class_name, registry_name in _MOCK_SCENARIOS
    )
    (pkg / "bootstrap.py").write_text(
        textwrap.dedent(
            """\
            from pyrit.registry import ScenarioRegistry

            {imports}


            def register() -> None:
                registry = ScenarioRegistry.get_registry_singleton()
            {registrations}
            """
        ).format(imports=imports, registrations=registrations),
        encoding="utf-8",
    )

    distinfo = src / f"{package}-0.0.1.dist-info"
    distinfo.mkdir(parents=True, exist_ok=True)
    (distinfo / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {package}\nVersion: 0.0.1\n", encoding="utf-8")
    (distinfo / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n", encoding="utf-8"
    )
    (distinfo / "RECORD").write_text("", encoding="utf-8")

    wheel = dest_dir / f"{package}-0.0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(src):
            for file_name in files:
                file_path = Path(root) / file_name
                archive.write(file_path, str(file_path.relative_to(src)))
    return wheel


def _registered_scenario_class_names() -> set[str]:
    """Return the class names of every scenario currently registered in ScenarioRegistry."""
    registry = ScenarioRegistry.get_registry_singleton()
    return {registry.get_class(name).__name__ for name in registry.get_class_names()}


def all_scenario_class_names(from_dirs: list[Path]) -> set[str]:
    """
    Walk source directories and collect the class name of every ``Scenario`` subclass.

    This is the expected-set builder: given the scenario source directories to cover, it
    enumerates the scenario classes defined there so the test can assert each is
    discovered after initialization.

    Args:
        from_dirs: Directories to walk recursively for ``Scenario`` subclasses.

    Returns:
        set[str]: The scenario class names found across the directories.
    """
    names: set[str] = set()
    for directory in from_dirs:
        for _stem, _path, cls in discover_in_directory(directory=directory, base_class=Scenario, recursive=True):
            names.add(cls.__name__)
    return names


def _scenario_class_names_under_package(package_prefix: str) -> set[str]:
    """
    Return class names of concrete ``Scenario`` subclasses whose module is under a package.

    Uses in-memory subclass enumeration (reliable after the plug-in has been imported),
    which sidesteps the standalone-import problems a filesystem walk can hit for a
    package that uses relative imports.

    Args:
        package_prefix: The plug-in's top-level package name.

    Returns:
        set[str]: Scenario class names owned by that package.
    """
    prefix = f"{package_prefix}."
    found: set[str] = set()
    seen: set[type] = set()
    stack: list[type] = list(Scenario.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        module = cls.__module__ or ""
        if not inspect.isabstract(cls) and (module == package_prefix or module.startswith(prefix)):
            found.add(cls.__name__)
    return found


@contextmanager
def _plugin_dir_env(*, plugin_dir: Path | None) -> Iterator[None]:
    """Set PLUGIN_DIR (extraction dir) for the duration of a load, then restore it."""
    values: dict[str, str] = {}
    if plugin_dir is not None:
        values["PLUGIN_DIR"] = str(plugin_dir)

    saved: dict[str, str | None] = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


@pytest.fixture
def plugin_sandbox() -> Iterator[None]:
    """Snapshot and restore global import + registry state so a load does not leak."""
    sys_path_snapshot = list(sys.path)
    modules_snapshot = set(sys.modules)
    provider_snapshot = dict(SeedDatasetProvider._registry)

    yield

    sys.path[:] = sys_path_snapshot
    for name in set(sys.modules) - modules_snapshot:
        del sys.modules[name]
    SeedDatasetProvider._registry.clear()
    SeedDatasetProvider._registry.update(provider_snapshot)
    ScenarioRegistry.reset_registry_singleton()


@pytest.mark.run_only_if_all_tests
async def test_built_wheel_scenarios_are_discovered(tmp_path: Path, plugin_sandbox: None) -> None:
    """A built wheel's scenarios are all registered in ScenarioRegistry after init."""
    package = f"mock_scenario_plugin_{uuid.uuid4().hex[:8]}"
    wheel = _build_scenario_plugin_wheel(tmp_path, package=package)

    registry = ScenarioRegistry.get_registry_singleton()
    before = set(registry.get_class_names())

    with _plugin_dir_env(plugin_dir=tmp_path / ".plugin"):
        await initialize_pyrit_async(IN_MEMORY, plugins=[PluginSpec(wheel=wheel)])

    registry = ScenarioRegistry.get_registry_singleton()
    after = set(registry.get_class_names())

    expected_registry_names = {registry_name for _stem, _cls, registry_name in _MOCK_SCENARIOS}
    # The plug-in adds exactly its scenarios and removes none of the built-ins.
    assert after - before == expected_registry_names
    assert before <= after

    # And the scenario classes themselves resolve to the plug-in's classes.
    expected_class_names = {class_name for _stem, class_name, _ in _MOCK_SCENARIOS}
    assert expected_class_names <= _registered_scenario_class_names()


@pytest.mark.run_only_if_all_tests
async def test_injected_wheel_scenarios_are_discovered(plugin_sandbox: None) -> None:
    """Every scenario in an out-of-band wheel is discovered after initialization.

    Skipped unless ``PLUGIN_TEST_WHEEL`` is set, so the committed test names no external
    package. Provide ``PLUGIN_TEST_SCENARIO_DIRS`` (preferred) or ``PLUGIN_TEST_PACKAGE``
    to tell the test which scenarios to expect.
    """
    wheel_env = os.getenv("PLUGIN_TEST_WHEEL")
    if not wheel_env:
        pytest.skip("PLUGIN_TEST_WHEEL is not set; skipping injected-wheel scenario discovery test.")

    wheel = Path(wheel_env).expanduser()
    assert wheel.is_file(), f"PLUGIN_TEST_WHEEL does not exist: {wheel}"

    scenario_dirs_env = os.getenv("PLUGIN_TEST_SCENARIO_DIRS")
    package = os.getenv("PLUGIN_TEST_PACKAGE")
    if not scenario_dirs_env and not package:
        pytest.skip("Set PLUGIN_TEST_SCENARIO_DIRS or PLUGIN_TEST_PACKAGE to define the expected scenarios.")

    with _plugin_dir_env(plugin_dir=None):
        await initialize_pyrit_async(IN_MEMORY, plugins=[PluginSpec(wheel=wheel, package=package)])

    found = _registered_scenario_class_names()

    if scenario_dirs_env:
        dirs = [Path(p) for p in scenario_dirs_env.split(os.pathsep) if p]
        expected = all_scenario_class_names(dirs)
    else:
        assert package is not None
        expected = _scenario_class_names_under_package(package)

    assert expected, "No plug-in scenarios were found to verify; check the injected wheel/scenario source."
    missing = expected - found
    assert not missing, f"Plug-in scenarios not discovered by ScenarioRegistry: {sorted(missing)}"
