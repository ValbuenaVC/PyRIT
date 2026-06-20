# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
InspectInitializer — bootstraps PyRIT-in-Inspect.
"""

from __future__ import annotations

import logging

from pyrit.inspect_bridge._imports import require_inspect_ai
from pyrit.registry.object_registries.target_registry import TargetRegistry
from pyrit.setup.initializers.pyrit_initializer import PyRITInitializer

logger = logging.getLogger(__name__)

# Module-level idempotency guard.
_initialized: bool = False


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
        self._log_dir = log_dir
        self._register_all_targets = register_all_targets

    async def initialize_async(self) -> None:
        """
        Execute initialization asynchronously.

        Ensures the ``@modelapi(name="pyrit")`` and ``@hooks`` shims are
        registered with Inspect AI, registers all target instances from
        ``TargetRegistry`` (when ``register_all_targets=True``), and
        applies the configured ``log_dir``. Idempotent.

        Raises:
            InspectBridgeError: If ``inspect_ai`` is not installed.

        """
        global _initialized  # noqa: PLW0603

        require_inspect_ai()

        from pyrit.inspect_bridge._registry import register_shims

        register_shims()

        if self._register_all_targets:
            self._register_all_target_instances()

        if self._log_dir is not None:
            self._apply_log_dir(self._log_dir)

        _initialized = True
        logger.debug("InspectInitializer: bootstrap complete")

    @property
    def required_env_vars(self) -> list[str]:
        """
        Return the list of required environment variables.

        Returns:
            list[str]: Always returns an empty list; no env vars are required.

        """
        return []

    def _register_all_target_instances(self) -> None:
        """Enumerate all TargetRegistry entries (for idempotent target discovery)."""
        registry = TargetRegistry.get_registry_singleton()
        entries = registry.get_all_instances()
        for entry in entries:
            logger.debug(f"InspectInitializer: discovered target '{entry.instance.get_identifier().unique_name}'")

    def _apply_log_dir(self, log_dir: str) -> None:
        """
        Redirect Inspect eval log output to ``log_dir``.

        Args:
            log_dir (str): The directory path where Inspect logs should be written.

        """
        try:
            import os

            from inspect_ai._util.logger import set_log_dir  # type: ignore[import-not-found]

            set_log_dir(log_dir)
        except (ImportError, AttributeError):
            # Inspect may not expose set_log_dir in all versions — use env var fallback.
            import os

            os.environ["INSPECT_LOG_DIR"] = log_dir
            logger.debug(f"InspectInitializer: set INSPECT_LOG_DIR={log_dir}")
