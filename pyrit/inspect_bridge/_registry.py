# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Registry shims for the Inspect AI model provider and hooks.

This module is referenced by the ``[project.entry-points.inspect_ai]`` entry
point in ``pyproject.toml`` so that Inspect AI discovers the PyRIT model
provider and hooks automatically. Heavy imports of ``inspect_ai`` are deferred
to the decorated functions/classes so that importing this module does not
trigger an ``import inspect_ai``.
"""

from __future__ import annotations

# Idempotency guard — set to True after the shims are applied.
_shims_registered: bool = False


def register_shims() -> None:
    """
    Register the PyRIT model-API and hooks shims with Inspect AI.

    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _shims_registered  # noqa: PLW0603
    if _shims_registered:
        return

    from inspect_ai.hooks._hooks import hooks
    from inspect_ai.model import modelapi

    from pyrit.inspect_bridge._memory_adapter import MemoryAdapter
    from pyrit.inspect_bridge._target_adapter import TargetToModelAdapter

    # Register the PyRIT model provider so `get_model("pyrit/...")` works.
    modelapi("pyrit")(TargetToModelAdapter)

    # Register the MemoryAdapter as an Inspect hooks subscriber.
    hooks(name="pyrit-memory-adapter", description="Persists Inspect transcripts to PyRIT CentralMemory")(
        MemoryAdapter
    )

    _shims_registered = True
