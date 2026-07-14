# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Integration test: plug-in scenarios load, register, instantiate, and execute after init.

This exercises the full public loading path end-to-end: point ``PLUGIN_WHEEL`` at a
built wheel, run ``initialize_pyrit_async``, and assert the wheel's scenarios are
registered in ``ScenarioRegistry``, construct cleanly, and drive through the public
execution pipeline.

Two cases:

* A self-contained case builds a small wheel at test time and verifies its scenarios
  are discovered. This always runs and guards the mechanism in public CI.
* An injected case loads a wheel supplied out-of-band via environment variables and
  verifies all of its scenarios are discovered, instantiate, and that at least one
  executes. It is skipped unless those variables are set, so the committed test depends
  on no specific external package. Point it at a real scenario wheel (for example in a
  downstream/private CI job) to guarantee every scenario that wheel ships is picked up::

      PLUGIN_TEST_WHEEL=/path/to/plugin.whl
      PLUGIN_TEST_PACKAGE=the_plugin_package          # enables package enumeration + instantiation
      PLUGIN_TEST_SCENARIO_DIRS=/path/to/scenarios    # optional; os.pathsep-separated
      PLUGIN_TEST_EXEC_SCENARIO=the.registry.name     # optional; enables the execution case
      ADVERSARIAL_CHAT_ENDPOINT=...                    # execution target + scorer endpoint
      ADVERSARIAL_CHAT_MODEL=...                       # optional model name
"""

import inspect
import os
import re
import subprocess
import sys
import textwrap
import uuid
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from pyrit.datasets.seed_datasets.seed_dataset_provider import SeedDatasetProvider
from pyrit.models import ScenarioResult, ScenarioRunState
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.registry import AttackTechniqueRegistry, ScenarioRegistry
from pyrit.registry.discovery import discover_in_directory
from pyrit.scenario.core import DatasetAttackConfiguration, Scenario
from pyrit.score import SelfAskTrueFalseScorer, TrueFalseQuestionPaths
from pyrit.setup import ConfigurationLoader, PluginFormat, PluginSpec

# (module stem, class name, registry name) for the scenarios the self-contained wheel ships.
_MOCK_SCENARIOS = [
    ("alpha", "MockAlphaScenario", "alpha"),
    ("beta", "MockBetaScenario", "beta"),
    ("gamma", "MockGammaScenario", "gamma"),
]

# Substring of the ValueError ``Scenario.run_async`` raises when an atomic attack's
# objective did not complete (for example a target refusal). The attack still ran through
# the public pipeline, so this outcome proves execution while a drift regression -- which
# surfaces a different exception type before or during the run -- still fails loudly.
_OBJECTIVES_INCOMPLETE_MARKER = "objectives incomplete"


async def _initialize_plugin_async(*, spec: PluginSpec, plugin_dir: Path | None) -> None:
    """Initialize PyRIT through the config-owned privileged plug-in path."""
    config = ConfigurationLoader(
        memory_db_type="in_memory",
        initializers=["technique"],
        env_files=[],
        silent=True,
        plugins=[spec.to_config()],
    )
    with _plugin_dir_env(plugin_dir=plugin_dir):
        await config.initialize_pyrit_async()


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


def _scenario_classes_under_package(package_prefix: str) -> dict[str, type[Scenario]]:
    """
    Return registered concrete ``Scenario`` subclasses owned by a package, keyed by registry name.

    Uses the post-load registry (reliable after the plug-in has been imported and
    bootstrapped), which sidesteps the standalone-import problems a filesystem walk can
    hit for a package that uses relative imports.

    Args:
        package_prefix: The plug-in's top-level package name.

    Returns:
        dict[str, type[Scenario]]: Registry name -> scenario class for that package.
    """
    prefix = f"{package_prefix}."
    registry = ScenarioRegistry.get_registry_singleton()
    found: dict[str, type[Scenario]] = {}
    for name in registry.get_class_names():
        cls = registry.get_class(name)
        module = cls.__module__ or ""
        if not inspect.isabstract(cls) and (module == package_prefix or module.startswith(prefix)):
            found[name] = cls
    return found


async def _execute_scenario_async(
    *, scenario_cls: type[Scenario], endpoint: str, model: str | None
) -> ScenarioResult | None:
    """
    Drive one plug-in scenario through the public initialize/run pipeline.

    Constructs an objective target and scorer from the supplied endpoint, initializes the
    scenario, and runs it. ``initialize_async`` must succeed and build at least one atomic
    attack; that is where dataset-config resolution drift would surface.

    Args:
        scenario_cls: The plug-in scenario class to execute.
        endpoint: Chat endpoint backing both the objective target and the scorer.
        model: Optional model name for the endpoint.

    Returns:
        ScenarioResult | None: The completed result, or ``None`` when the run finished with
        incomplete objectives (the attack ran end-to-end but the objective was not achieved).
    """
    target = OpenAIChatTarget(endpoint=endpoint, model_name=model)
    scorer = SelfAskTrueFalseScorer(
        chat_target=target,
        true_false_question_path=TrueFalseQuestionPaths.TASK_ACHIEVED_REFINED.value,
    )

    try:
        scenario = scenario_cls(objective_scorer=scorer, fast_mode=True)  # type: ignore[ty:missing-argument, ty:unknown-argument]
    except TypeError:
        scenario = scenario_cls(objective_scorer=scorer)  # type: ignore[ty:missing-argument]

    dataset_config: DatasetAttackConfiguration | None = None
    required_datasets = getattr(scenario_cls, "required_datasets", None)
    if callable(required_datasets):
        names = list(required_datasets())
        if names:
            dataset_config = DatasetAttackConfiguration(dataset_names=names, max_dataset_size=1)

    args: dict[str, Any] = {"objective_target": target, "max_concurrency": 1}
    if dataset_config is not None:
        args["dataset_config"] = dataset_config
    scenario.set_params_from_args(args=args)
    await scenario.initialize_async()
    assert scenario.atomic_attack_count >= 1, "Scenario built no atomic attacks during initialization."

    try:
        return await scenario.run_async()
    except ValueError as exc:
        if _OBJECTIVES_INCOMPLETE_MARKER in str(exc):
            return None
        raise


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
def plugin_dir(tmp_path: Path) -> Path:
    """The directory the loader extracts wheels into for a test."""
    return tmp_path / ".plugin"


@pytest.fixture
def build_mock_wheel(tmp_path: Path) -> Callable[[str], Path]:
    """A builder for the self-contained scenario wheel, parameterized by package name."""

    def build(package: str) -> Path:
        return _build_scenario_plugin_wheel(tmp_path, package=package)

    return build


@pytest.fixture
def all_scenarios() -> Callable[[list[Path]], set[str]]:
    """The expected-set builder: scenario class names defined under the given source dirs."""
    return all_scenario_class_names


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
    AttackTechniqueRegistry.reset_registry_singleton()
    ScenarioRegistry.reset_registry_singleton()


@pytest.mark.run_only_if_all_tests
async def test_built_wheel_scenarios_are_discovered(
    plugin_sandbox: None,
    build_mock_wheel: Callable[[str], Path],
    plugin_dir: Path,
) -> None:
    """A built wheel's scenarios are all registered in ScenarioRegistry after init."""
    package = f"mock_scenario_plugin_{uuid.uuid4().hex[:8]}"
    wheel = build_mock_wheel(package)

    registry = ScenarioRegistry.get_registry_singleton()
    before = set(registry.get_class_names())

    await _initialize_plugin_async(
        spec=PluginSpec(
            name=package,
            format=PluginFormat.WHEEL,
            wheel=wheel,
            package=package,
        ),
        plugin_dir=plugin_dir,
    )

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
async def test_injected_wheel_scenarios_are_discovered(
    plugin_sandbox: None,
    all_scenarios: Callable[[list[Path]], set[str]],
) -> None:
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

    await _initialize_plugin_async(
        spec=PluginSpec(
            name=package or "injected_plugin",
            format=PluginFormat.WHEEL,
            wheel=wheel,
            package=package,
        ),
        plugin_dir=None,
    )

    found = _registered_scenario_class_names()

    if scenario_dirs_env:
        dirs = [Path(p) for p in scenario_dirs_env.split(os.pathsep) if p]
        expected = all_scenarios(dirs)
    else:
        assert package is not None
        expected = {cls.__name__ for cls in _scenario_classes_under_package(package).values()}

    assert expected, "No plug-in scenarios were found to verify; check the injected wheel/scenario source."
    missing = expected - found
    assert not missing, f"Plug-in scenarios not discovered by ScenarioRegistry: {sorted(missing)}"


@pytest.mark.run_only_if_all_tests
async def test_injected_wheel_scenarios_instantiate(plugin_sandbox: None) -> None:
    """Every scenario in an out-of-band wheel constructs cleanly after initialization.

    Skipped unless ``PLUGIN_TEST_WHEEL`` and ``PLUGIN_TEST_PACKAGE`` are set.
    """
    wheel_env = os.getenv("PLUGIN_TEST_WHEEL")
    package = os.getenv("PLUGIN_TEST_PACKAGE")
    if not wheel_env or not package:
        pytest.skip("Set PLUGIN_TEST_WHEEL and PLUGIN_TEST_PACKAGE to run the instantiation test.")

    wheel = Path(wheel_env).expanduser()
    assert wheel.is_file(), f"PLUGIN_TEST_WHEEL does not exist: {wheel}"

    await _initialize_plugin_async(
        spec=PluginSpec(
            name=package,
            format=PluginFormat.WHEEL,
            wheel=wheel,
            package=package,
        ),
        plugin_dir=None,
    )

    classes = _scenario_classes_under_package(package)
    assert classes, f"No plug-in scenarios registered under package {package!r}."

    failures: dict[str, str] = {}
    for name, cls in sorted(classes.items()):
        try:
            instance = cls()  # type: ignore[ty:missing-argument]
            assert isinstance(instance, Scenario)
        except Exception as exc:  # noqa: BLE001
            failures[name] = f"{type(exc).__name__}: {exc}"

    assert not failures, f"Plug-in scenarios failed to instantiate: {failures}"


@pytest.mark.run_only_if_all_tests
async def test_injected_wheel_scenario_executes(plugin_sandbox: None) -> None:
    """One named scenario from an out-of-band wheel runs through the public pipeline.

    Skipped unless ``PLUGIN_TEST_WHEEL``, ``PLUGIN_TEST_PACKAGE``, and
    ``PLUGIN_TEST_EXEC_SCENARIO`` are set, and an ``ADVERSARIAL_CHAT_ENDPOINT`` is
    configured (it may come from the loaded ``.env``, so it is checked after init).
    """
    wheel_env = os.getenv("PLUGIN_TEST_WHEEL")
    package = os.getenv("PLUGIN_TEST_PACKAGE")
    exec_scenario = os.getenv("PLUGIN_TEST_EXEC_SCENARIO")
    if not wheel_env or not package or not exec_scenario:
        pytest.skip("Set PLUGIN_TEST_WHEEL, PLUGIN_TEST_PACKAGE, and PLUGIN_TEST_EXEC_SCENARIO to run the exec test.")

    wheel = Path(wheel_env).expanduser()
    assert wheel.is_file(), f"PLUGIN_TEST_WHEEL does not exist: {wheel}"

    await _initialize_plugin_async(
        spec=PluginSpec(
            name=package,
            format=PluginFormat.WHEEL,
            wheel=wheel,
            package=package,
        ),
        plugin_dir=None,
    )

    endpoint = os.getenv("ADVERSARIAL_CHAT_ENDPOINT")
    if not endpoint:
        pytest.skip("ADVERSARIAL_CHAT_ENDPOINT is not configured; skipping plug-in scenario execution.")

    registry = ScenarioRegistry.get_registry_singleton()
    assert exec_scenario in registry.get_class_names(), (
        f"PLUGIN_TEST_EXEC_SCENARIO {exec_scenario!r} is not registered; "
        f"available: {sorted(registry.get_class_names())}"
    )
    scenario_cls = registry.get_class(exec_scenario)

    result = await _execute_scenario_async(
        scenario_cls=scenario_cls,
        endpoint=endpoint,
        model=os.getenv("ADVERSARIAL_CHAT_MODEL"),
    )

    if result is not None:
        assert result.scenario_run_state == ScenarioRunState.COMPLETED
        assert result.attack_results, "Completed scenario run produced no attack results."


# ---------------------------------------------------------------------------
# Scanner integration
# ---------------------------------------------------------------------------


def _write_scanner_config(*, tmp_path: Path, source: Path, plugin_name: str) -> Path:
    config = tmp_path / f"{plugin_name}.pyrit_conf"
    config.write_text(
        textwrap.dedent(
            f"""\
            memory_db_type: in_memory
            initializers:
              - target
              - scorer
              - technique
            plugins:
              - name: {plugin_name}
                format: source
                source: {source.as_posix()}
            """
        ),
        encoding="utf-8",
    )
    return config


def _run_scanner_with_fresh_backend(*, config: Path) -> subprocess.CompletedProcess[str]:
    base = [sys.executable, "-m", "pyrit.cli.pyrit_scan", "--config-file", str(config)]
    subprocess.run([*base, "--stop-server"], capture_output=True, text=True, check=False, timeout=30)
    try:
        return subprocess.run(
            [*base, "--start-server", "--list-scenarios"],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
    finally:
        subprocess.run([*base, "--stop-server"], capture_output=True, text=True, check=False, timeout=30)


@pytest.mark.run_only_if_all_tests
def test_private_scenario_registers_from_scanner(tmp_path: Path) -> None:
    """A source Scenario appears through the stock scanner catalog."""
    source = tmp_path / "private_scanner_scenario.py"
    source.write_text(
        """from pyrit.models import SeedObjective
from pyrit.scenario import DatasetAttackConfiguration, Scenario, ScenarioTechnique
from pyrit.score import SubStringScorer

class PrivateTechnique(ScenarioTechnique):
    ALL = ("all", {"all"})
    DIRECT = ("direct", {"direct"})

class PrivateScannerScenario(Scenario):
    VERSION = 1

    def __init__(self, *, scenario_result_id=None):
        super().__init__(
            version=self.VERSION,
            technique_class=PrivateTechnique,
            default_technique=PrivateTechnique.ALL,
            default_dataset_config=DatasetAttackConfiguration(
                seeds=[SeedObjective(value="integration objective")]
            ),
            objective_scorer=SubStringScorer(substring="integration"),
            scenario_result_id=scenario_result_id,
        )

    async def _build_atomic_attacks_async(self, *, context):
        return []
""",
        encoding="utf-8",
    )
    config = _write_scanner_config(
        tmp_path=tmp_path,
        source=source,
        plugin_name="private_scanner_scenario",
    )

    proc = _run_scanner_with_fresh_backend(config=config)

    assert "private_scanner_scenario" in proc.stdout
    assert "PrivateScannerScenario" in proc.stdout


@pytest.mark.run_only_if_all_tests
def test_private_attack_technique_registers_from_scanner(tmp_path: Path) -> None:
    """A source technique applicable to RapidResponse appears in scanner metadata."""
    source = tmp_path / "private_scanner_technique.py"
    source.write_text(
        """from pyrit.executor.attack import PromptSendingAttack
from pyrit.scenario import AttackTechniqueFactory

PRIVATE = AttackTechniqueFactory(
    name="private_scanner_technique",
    attack_class=PromptSendingAttack,
    technique_tags=["single_turn", "scenario:airt.rapid_response"],
)
""",
        encoding="utf-8",
    )
    config = _write_scanner_config(
        tmp_path=tmp_path,
        source=source,
        plugin_name="private_scanner_technique",
    )

    proc = _run_scanner_with_fresh_backend(config=config)

    match = re.search(
        r"\n  airt\.rapid_response\n(?P<body>.*?)(?=\n  [a-z][a-z0-9_.]*\n|\n={80})",
        proc.stdout,
        flags=re.DOTALL,
    )
    assert match is not None, proc.stdout
    assert "private_scanner_technique" in match.group("body")
