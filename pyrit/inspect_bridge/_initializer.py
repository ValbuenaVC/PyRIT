# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
InspectInitializer — bootstraps PyRIT-in-Inspect.
"""

from __future__ import annotations

from pyrit.setup.initializers.pyrit_initializer import PyRITInitializer


class InspectInitializer(PyRITInitializer):
    """
    Bootstraps PyRIT-in-Inspect.

    Ensures the ``@modelapi(name="pyrit")`` provider and the ``@hooks``
    ``MemoryAdapter`` are registered (via the package's inspect_ai entry points /
    decorated shims in ``_registry.py``), registers ``TargetRegistry`` instances
    as resolvable ``pyrit/<name>`` models, and redirects Inspect ``.eval`` logs
    into PyRIT's log dir. Idempotent.
    """

    def __init__(self, *, log_dir: str | None = None, register_all_targets: bool = True) -> None:
        """
        Initialize the InspectInitializer.

        Args:
            log_dir (str | None): Optional directory for Inspect eval log output.
                Defaults to PyRIT's configured log directory.
            register_all_targets (bool): When True, all instances registered in
                ``TargetRegistry`` are made available as ``pyrit/<name>`` Inspect models.

        """
        super().__init__()
        raise NotImplementedError

    async def initialize_async(self) -> None:
        """Execute initialization asynchronously."""
        raise NotImplementedError

    @property
    def required_env_vars(self) -> list[str]:
        """
        Return the list of required environment variables.

        Returns:
            list[str]: Always returns an empty list; no env vars are required.

        """
        raise NotImplementedError
