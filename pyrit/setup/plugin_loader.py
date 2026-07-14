# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Activate a configured plug-in by running its ``PyRITInitializer`` first.

A plug-in is a config-declared pointer to a ``PyRITInitializer`` that lives outside the
PyRIT tree. ``ConfigurationLoader`` prepends this privileged initializer ahead of the
user-configured ones, so the plug-in's components exist before any later initializer or
lazy catalog consumer runs. The plug-in initializer itself owns registration of every
scenario, attack technique, dataset, and default target it wants discoverable; PyRIT does
no component discovery here.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from typing import TYPE_CHECKING

from pyrit.exceptions import PluginLoadError, PluginSourceNotFoundError
from pyrit.setup.pyrit_initializer import PyRITInitializer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyrit.setup.plugin_spec import PluginSpec

logger = logging.getLogger(__name__)

_REMEDIATION = "Fix or remove the plug-in entry from .pyrit_conf, then restart PyRIT."


class PluginInitializer(PyRITInitializer):
    """Privileged, config-owned initializer that runs one plug-in's ``PyRITInitializer`` first."""

    def __init__(self, *, plugins: Sequence[PluginSpec]) -> None:
        """
        Initialize the privileged plug-in loader.

        Args:
            plugins (Sequence[PluginSpec]): The normalized plug-in declarations.

        Raises:
            ValueError: If anything other than exactly one plug-in is supplied.
        """
        super().__init__()
        if len(plugins) != 1:
            raise ValueError("PluginInitializer requires exactly one plug-in.")
        self._spec = plugins[0]

    @property
    def plugins(self) -> list[PluginSpec]:
        """The normalized plug-in declarations."""
        return [self._spec]

    async def initialize_async(self) -> None:
        """
        Run the configured plug-in's initializer before user-configured initializers.

        Raises:
            PluginLoadError: If the source cannot be prepared, the initializer cannot be
                resolved, or the initializer raises. Fails closed — a restart is required
                after fixing the configuration or artifact.
        """
        initializer = await asyncio.to_thread(self._resolve_initializer)
        try:
            await initializer.initialize_async()
        except Exception as exc:
            raise PluginLoadError(f"Plug-in '{self._spec.name}' initializer failed: {exc} {_REMEDIATION}") from exc
        logger.info("Activated plug-in '%s' via %s.", self._spec.name, self._spec.initializer)

    def _resolve_initializer(self) -> PyRITInitializer:
        """
        Anchor the source root on ``sys.path`` and import the configured initializer.

        Returns:
            PyRITInitializer: A fresh instance of the resolved initializer subclass.

        Raises:
            PluginSourceNotFoundError: If the source path does not exist.
            PluginLoadError: If the dotted initializer cannot be imported or is not a
                ``PyRITInitializer`` subclass.
        """
        spec = self._spec
        if not spec.source.exists():
            raise PluginSourceNotFoundError(f"Plug-in '{spec.name}' source path does not exist: {spec.source}")

        root = spec.source if spec.source.is_dir() else spec.source.parent
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        module_path, _, class_name = spec.initializer.rpartition(".")
        try:
            module = importlib.import_module(module_path)
            initializer_cls = getattr(module, class_name)
        except Exception as exc:
            raise PluginLoadError(
                f"Could not import plug-in '{spec.name}' initializer '{spec.initializer}' from {root}: "
                f"{exc} {_REMEDIATION}"
            ) from exc

        if not (isinstance(initializer_cls, type) and issubclass(initializer_cls, PyRITInitializer)):
            raise PluginLoadError(
                f"Plug-in '{spec.name}' initializer '{spec.initializer}' is not a PyRITInitializer subclass. "
                f"{_REMEDIATION}"
            )
        return initializer_cls()
