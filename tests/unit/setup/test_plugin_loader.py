# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the initializer-pointing ``PluginInitializer``.

A plug-in resolves to a ``PyRITInitializer`` reached by dotted path from a source root.
The loader anchors the root on ``sys.path``, imports the initializer, and runs it. The
initializer itself owns all registration; the loader only orchestrates and fails closed.
"""

import sys
import textwrap
from pathlib import Path

import pytest

from pyrit.exceptions import PluginLoadError, PluginSourceNotFoundError
from pyrit.registry import AttackTechniqueRegistry
from pyrit.setup import PluginSpec
from pyrit.setup.plugin_loader import PluginInitializer

_TEST_PACKAGE_ROOTS = {"mock_plugin", "notinit"}


def _write_plugin_package(root: Path, *, technique_name: str) -> None:
    """Write a source package whose initializer registers one private technique."""
    pkg = root / "mock_plugin"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "bootstrap.py").write_text(
        textwrap.dedent(
            f"""
            from pyrit.executor.attack import PromptSendingAttack
            from pyrit.registry import AttackTechniqueRegistry
            from pyrit.scenario.core import AttackTechniqueFactory
            from pyrit.setup.pyrit_initializer import PyRITInitializer


            class MockInitializer(PyRITInitializer):
                \"\"\"Register one private attack technique.\"\"\"

                async def initialize_async(self) -> None:
                    AttackTechniqueRegistry.get_registry_singleton().register_from_factories(
                        [AttackTechniqueFactory(name="{technique_name}", attack_class=PromptSendingAttack)]
                    )
            """
        ),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _isolate_import_state():
    original_path = list(sys.path)
    original_modules = set(sys.modules)
    AttackTechniqueRegistry.reset_registry_singleton()
    yield
    AttackTechniqueRegistry.reset_registry_singleton()
    sys.path[:] = original_path
    for name in list(sys.modules):
        if name not in original_modules and name.split(".", 1)[0] in _TEST_PACKAGE_ROOTS:
            del sys.modules[name]


async def test_runs_configured_initializer(tmp_path: Path) -> None:
    _write_plugin_package(tmp_path, technique_name="operation_alpha")
    spec = PluginSpec(name="alpha", source=tmp_path, initializer="mock_plugin.bootstrap.MockInitializer")

    await PluginInitializer(plugins=[spec]).initialize_async()

    assert "operation_alpha" in AttackTechniqueRegistry.get_registry_singleton().get_factories()


async def test_missing_source_fails_closed(tmp_path: Path) -> None:
    spec = PluginSpec(name="p", source=tmp_path / "nope", initializer="mock_plugin.bootstrap.MockInitializer")
    with pytest.raises(PluginSourceNotFoundError):
        await PluginInitializer(plugins=[spec]).initialize_async()


async def test_unresolvable_initializer_fails_closed(tmp_path: Path) -> None:
    _write_plugin_package(tmp_path, technique_name="operation_beta")
    spec = PluginSpec(name="p", source=tmp_path, initializer="mock_plugin.bootstrap.DoesNotExist")
    with pytest.raises(PluginLoadError):
        await PluginInitializer(plugins=[spec]).initialize_async()


async def test_non_initializer_target_fails_closed(tmp_path: Path) -> None:
    pkg = tmp_path / "notinit"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("class NotAnInitializer:\n    pass\n", encoding="utf-8")
    spec = PluginSpec(name="p", source=tmp_path, initializer="notinit.NotAnInitializer")
    with pytest.raises(PluginLoadError, match="not a PyRITInitializer"):
        await PluginInitializer(plugins=[spec]).initialize_async()


async def test_initializer_that_raises_fails_closed(tmp_path: Path) -> None:
    pkg = tmp_path / "notinit"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        textwrap.dedent(
            """
            from pyrit.setup.pyrit_initializer import PyRITInitializer


            class Boom(PyRITInitializer):
                async def initialize_async(self) -> None:
                    raise RuntimeError("boom")
            """
        ),
        encoding="utf-8",
    )
    spec = PluginSpec(name="p", source=tmp_path, initializer="notinit.Boom")
    with pytest.raises(PluginLoadError, match="initializer failed"):
        await PluginInitializer(plugins=[spec]).initialize_async()


def test_requires_exactly_one_plugin(tmp_path: Path) -> None:
    spec = PluginSpec(name="p", source=tmp_path, initializer="m.C")
    with pytest.raises(ValueError):
        PluginInitializer(plugins=[spec, spec])
