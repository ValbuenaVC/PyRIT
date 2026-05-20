# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Regression coverage for ``discover_in_package`` resilience to non-class type aliases."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

from pyrit.registry.discovery import discover_in_package

if TYPE_CHECKING:
    from pathlib import Path


def _write_module(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body), encoding="utf-8")


@pytest.fixture
def poisoned_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, type]:
    """
    Build a synthetic package with a module that exposes a parameterized
    ``Callable`` alias alongside a real concrete subclass of a synthetic base.
    The base class lives inside the fixture package so the synthetic module
    can import it without depending on the ``tests`` namespace being importable.
    """
    pkg_root = tmp_path / "_discovery_pkg"
    pkg_root.mkdir()
    _write_module(
        pkg_root / "__init__.py",
        """
        class DiscoveryBase:
            pass
        """,
    )

    _write_module(
        pkg_root / "good_module.py",
        """
        from collections.abc import Callable
        from _discovery_pkg import DiscoveryBase

        # Parameterized Callable type alias — inspect.isclass(alias) is True
        # but issubclass(alias, anything) raises TypeError. Before the fix this
        # poisoned the rest of the module's discovery.
        Poisoned = Callable[[int], str]

        class _RealConcreteAfter(DiscoveryBase):
            pass
        """,
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib

    pkg = importlib.import_module("_discovery_pkg")
    return pkg_root, "_discovery_pkg", pkg.DiscoveryBase


def test_discovery_skips_non_class_aliases(poisoned_package):
    pkg_root, pkg_name, base_cls = poisoned_package
    discovered = list(
        discover_in_package(
            package_path=pkg_root,
            package_name=pkg_name,
            base_class=base_cls,
            recursive=False,
        )
    )
    discovered_names = {cls.__name__ for _, cls in discovered}
    assert "_RealConcreteAfter" in discovered_names


def test_text_adaptive_registers_in_scenario_registry():
    """
    End-to-end regression: ``TextAdaptive`` lives in a module that exposes a
    parameterized Callable alias (``ContextExtractor``) at module scope. It
    must still appear in the registry after discovery.
    """
    from pyrit.registry.class_registries.scenario_registry import ScenarioRegistry

    registry = ScenarioRegistry()
    registry._ensure_discovered()
    names = sorted(registry._class_entries.keys())
    assert "adaptive.text_adaptive" in names
