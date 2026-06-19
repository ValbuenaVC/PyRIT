# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Inspect AI bridge for PyRIT.

This package provides adapters that let PyRIT run inside Inspect AI's evaluation
framework. All ``inspect_ai`` imports inside this package are deferred
(function/method-local) so that ``import pyrit`` never imports ``inspect_ai``.

Public API (also enumerated in ``__all__``):

- ``InspectInitializer`` — bootstraps PyRIT-in-Inspect.
- ``TargetToModelAdapter`` — wraps a ``PromptTarget`` as an Inspect model.
- ``DatasetAdapter`` — converts ``SeedAttackGroup`` lists to Inspect datasets.
- ``AttackToSolverAdapter`` — wraps an ``AttackStrategy`` as an Inspect solver.
- ``MemoryAdapter`` — persists Inspect transcripts to PyRIT ``CentralMemory``.
- ``InspectTaskFactory`` — the single entry point for creating/running Inspect evals.
- ``PYRIT_MODEL_PROVIDER`` — the Inspect model-provider prefix (``"pyrit"``).
- ``InspectBridgeError`` — base exception for bridge failures.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from pyrit.inspect_bridge.errors import InspectBridgeError

PYRIT_MODEL_PROVIDER: str = "pyrit"

__all__ = [
    "InspectInitializer",
    "TargetToModelAdapter",
    "DatasetAdapter",
    "AttackToSolverAdapter",
    "MemoryAdapter",
    "InspectTaskFactory",
    "PYRIT_MODEL_PROVIDER",
    "InspectBridgeError",
]

if TYPE_CHECKING:
    from pyrit.inspect_bridge._initializer import InspectInitializer
    from pyrit.inspect_bridge._memory_adapter import MemoryAdapter
    from pyrit.inspect_bridge._solver_adapter import AttackToSolverAdapter
    from pyrit.inspect_bridge._target_adapter import TargetToModelAdapter
    from pyrit.inspect_bridge._task_factory import InspectTaskFactory
    from pyrit.inspect_bridge._dataset_adapter import DatasetAdapter

_MODULE_MAP: dict[str, str] = {
    "InspectInitializer": "pyrit.inspect_bridge._initializer",
    "TargetToModelAdapter": "pyrit.inspect_bridge._target_adapter",
    "DatasetAdapter": "pyrit.inspect_bridge._dataset_adapter",
    "AttackToSolverAdapter": "pyrit.inspect_bridge._solver_adapter",
    "MemoryAdapter": "pyrit.inspect_bridge._memory_adapter",
    "InspectTaskFactory": "pyrit.inspect_bridge._task_factory",
}


def __getattr__(name: str) -> object:
    if name in _MODULE_MAP:
        module = importlib.import_module(_MODULE_MAP[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
