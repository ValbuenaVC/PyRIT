# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Static plug-in configuration model shared by setup entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyrit.models import validate_registry_name


@dataclass(frozen=True)
class PluginSpec:
    """A normalized plug-in declaration: a source root plus a dotted initializer."""

    name: str
    source: Path
    initializer: str

    @classmethod
    def from_config(cls, entry: dict[str, Any], *, base_dir: Path | None = None) -> PluginSpec:
        """
        Normalize one explicit ``.pyrit_conf`` plug-in entry.

        Args:
            entry (dict[str, Any]): The YAML plug-in mapping.
            base_dir (Path | None): Directory used to resolve a relative ``source`` path.

        Returns:
            PluginSpec: The normalized plug-in declaration.

        Raises:
            ValueError: If the mapping is malformed.
        """
        if not isinstance(entry, dict):
            raise ValueError(f"Plug-in entry must be a mapping, got {type(entry).__name__}.")

        allowed = {"name", "source", "initializer"}
        unexpected = set(entry) - allowed
        if unexpected:
            raise ValueError(f"Plug-in entry has unexpected key(s): {sorted(unexpected)}. Allowed: {sorted(allowed)}.")

        name = entry.get("name")
        if not isinstance(name, str):
            raise ValueError("Plug-in entry requires a string 'name'.")
        validate_registry_name(name)

        source = entry.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError("Plug-in entry requires a non-empty string 'source' path.")

        initializer = entry.get("initializer")
        if not isinstance(initializer, str) or "." not in initializer:
            raise ValueError(
                "Plug-in entry requires a dotted 'initializer' path to a PyRITInitializer subclass "
                "(e.g. 'my_package.setup.MyInitializer')."
            )

        path = Path(source).expanduser()
        if not path.is_absolute():
            path = (base_dir or Path.cwd()) / path
        path = path.resolve()

        return cls(name=name, source=path, initializer=initializer)

    def to_config(self) -> dict[str, str]:
        """
        Serialize this spec to the explicit YAML-style mapping.

        Returns:
            dict[str, str]: The normalized configuration mapping.
        """
        return {"name": self.name, "source": str(self.source), "initializer": self.initializer}
