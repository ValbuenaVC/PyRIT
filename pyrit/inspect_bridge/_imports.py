# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Deferred import helper for inspect_ai.

All imports of inspect_ai within the inspect_bridge package MUST go through
this module or be placed inside function/method bodies. This ensures that
``import pyrit`` never triggers an ``import inspect_ai``, keeping the core
package fast and dependency-free for users who do not install the ``inspect`` extra.
"""

from __future__ import annotations


def require_inspect_ai() -> None:
    """
    Verify that inspect_ai is importable.

    Raises:
        InspectBridgeError: If ``inspect_ai`` is not installed, with a hint to
            install the ``inspect`` extra.

    """
    # Import here so the error class itself doesn't pull in inspect_ai.
    from pyrit.inspect_bridge.errors import InspectBridgeError

    try:
        import inspect_ai  # noqa: F401
    except ImportError as exc:
        raise InspectBridgeError(
            message=(
                "inspect_ai is not installed. "
                "Install the 'inspect' extra: pip install 'pyrit[inspect]'. "
                "Then run InspectInitializer to register the PyRIT model provider."
            )
        ) from exc
