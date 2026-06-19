# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Import guard tests for the inspect_bridge package.

Verifies that importing pyrit and pyrit.inspect_bridge never triggers
an import of inspect_ai, keeping the core package fast and dependency-free.
"""

import subprocess
import sys


def test_import_pyrit_does_not_import_inspect_ai() -> None:
    """``import pyrit`` must not trigger ``import inspect_ai``."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pyrit; import sys; assert 'inspect_ai' not in sys.modules, "
            "'inspect_ai was imported by import pyrit'",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"import pyrit triggered inspect_ai: {result.stderr}"


def test_import_inspect_bridge_does_not_import_inspect_ai() -> None:
    """``import pyrit.inspect_bridge`` must not trigger ``import inspect_ai``."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pyrit.inspect_bridge; import sys; "
            "assert 'inspect_ai' not in sys.modules, "
            "'inspect_ai was imported by import pyrit.inspect_bridge'",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"import pyrit.inspect_bridge triggered inspect_ai: {result.stderr}"


def test_inspect_bridge_all_exports_declared() -> None:
    """``__all__`` must list the eight public names from the frozen contract."""
    import pyrit.inspect_bridge as bridge

    expected = {
        "InspectInitializer",
        "TargetToModelAdapter",
        "DatasetAdapter",
        "AttackToSolverAdapter",
        "MemoryAdapter",
        "InspectTaskFactory",
        "PYRIT_MODEL_PROVIDER",
        "InspectBridgeError",
    }
    assert set(bridge.__all__) == expected


def test_pyrit_model_provider_constant() -> None:
    """``PYRIT_MODEL_PROVIDER`` must equal ``"pyrit"``."""
    from pyrit.inspect_bridge import PYRIT_MODEL_PROVIDER

    assert PYRIT_MODEL_PROVIDER == "pyrit"


def test_inspect_bridge_error_is_importable_without_inspect_ai() -> None:
    """``InspectBridgeError`` must be importable without inspect_ai installed."""
    from pyrit.inspect_bridge import InspectBridgeError
    from pyrit.exceptions.exception_classes import PyritException

    assert issubclass(InspectBridgeError, PyritException)


def test_inspect_bridge_error_default_args() -> None:
    """``InspectBridgeError`` must accept keyword-only args with correct defaults."""
    from pyrit.inspect_bridge import InspectBridgeError

    err = InspectBridgeError()
    assert err.status_code == 500
    assert err.message == "Inspect bridge error"


def test_inspect_bridge_error_custom_args() -> None:
    """``InspectBridgeError`` must propagate custom message and status_code."""
    from pyrit.inspect_bridge import InspectBridgeError

    err = InspectBridgeError(message="Did you run InspectInitializer?", status_code=503)
    assert err.status_code == 503
    assert "InspectInitializer" in err.message


def test_inspect_extra_declared_in_pyproject() -> None:
    """``[project.optional-dependencies] inspect`` must be declared in pyproject.toml."""
    import importlib.util
    import pathlib
    import re

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    pyproject = repo_root / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")

    assert re.search(r"^\s*\[project\.optional-dependencies\]", content, re.MULTILINE), (
        "pyproject.toml missing [project.optional-dependencies]"
    )
    assert "inspect-ai" in content, "inspect-ai not found in pyproject.toml extras"


def test_entry_point_declared_in_pyproject() -> None:
    """``[project.entry-points.inspect_ai]`` must declare ``pyrit``."""
    import pathlib
    import re

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    pyproject = repo_root / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")

    assert re.search(r"\[project\.entry-points\.inspect_ai\]", content), (
        "pyproject.toml missing [project.entry-points.inspect_ai]"
    )
    assert re.search(r'pyrit\s*=\s*"pyrit\.inspect_bridge\._registry"', content), (
        "entry-point pyrit -> pyrit.inspect_bridge._registry not found"
    )
