# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Static plug-in configuration models shared by setup entry points."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pyrit.models import validate_registry_name


class PluginFormat(str, Enum):
    """The supported plug-in artifact formats."""

    SOURCE = "source"
    WHEEL = "wheel"


@dataclass(frozen=True)
class PluginSpec:
    """A normalized plug-in artifact declaration."""

    wheel: Path | None = None
    package: str | None = None
    name: str | None = None
    format: PluginFormat | None = None
    source: Path | None = None

    def __post_init__(self) -> None:
        """
        Infer legacy direct-construction fields while the loader is refactored.

        Raises:
            ValueError: If the artifact fields are missing, conflicting, or inconsistent
                with the explicit format.
        """
        if (self.source is None) == (self.wheel is None):
            raise ValueError("PluginSpec requires exactly one of 'source' or 'wheel'.")

        inferred_format = PluginFormat.SOURCE if self.source is not None else PluginFormat.WHEEL
        if self.format is not None and self.format is not inferred_format:
            raise ValueError(f"Plug-in format '{self.format.value}' does not match its artifact field.")
        object.__setattr__(self, "format", inferred_format)

        if self.name is None:
            path = self.source or self.wheel
            assert path is not None
            inferred_name = self.package.split(".", 1)[0] if self.package else path.stem.replace("-", "_")
            object.__setattr__(self, "name", inferred_name)

    @property
    def artifact_path(self) -> Path:
        """The normalized source or wheel artifact path."""
        path = self.source or self.wheel
        assert path is not None
        return path

    @classmethod
    def from_config(cls, entry: dict[str, Any], *, base_dir: Path | None = None) -> PluginSpec:
        """
        Normalize one explicit ``.pyrit_conf`` plug-in entry.

        Args:
            entry (dict[str, Any]): The YAML plug-in mapping.
            base_dir (Path | None): Directory used to resolve relative artifact paths.

        Returns:
            PluginSpec: The normalized plug-in declaration.

        Raises:
            ValueError: If the mapping is malformed or inconsistent.
        """
        if not isinstance(entry, dict):
            raise ValueError(f"Plug-in entry must be a mapping, got {type(entry).__name__}.")

        allowed = {"name", "format", "source", "wheel", "package"}
        unexpected = set(entry) - allowed
        if unexpected:
            raise ValueError(f"Plug-in entry has unexpected key(s): {sorted(unexpected)}.")

        name = entry.get("name")
        if not isinstance(name, str):
            raise ValueError("Plug-in entry requires a string 'name'.")
        validate_registry_name(name)

        try:
            plugin_format = PluginFormat(entry.get("format"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Plug-in entry 'format' must be 'source' or 'wheel'.") from exc

        source = entry.get("source")
        wheel = entry.get("wheel")
        if (source is None) == (wheel is None):
            raise ValueError("Plug-in entry requires exactly one of 'source' or 'wheel'.")
        artifact_key = plugin_format.value
        artifact = entry.get(artifact_key)
        other_key = PluginFormat.WHEEL.value if plugin_format is PluginFormat.SOURCE else PluginFormat.SOURCE.value
        if not isinstance(artifact, str) or entry.get(other_key) is not None:
            raise ValueError(f"Plug-in format '{plugin_format.value}' requires only the '{artifact_key}' field.")

        package = entry.get("package")
        if package is not None and (
            not isinstance(package, str) or not all(part.isidentifier() for part in package.split("."))
        ):
            raise ValueError("Plug-in 'package' must be a dotted Python identifier.")

        path = Path(artifact).expanduser()
        if not path.is_absolute():
            path = (base_dir or Path.cwd()) / path
        path = path.resolve()

        return cls(
            name=name,
            format=plugin_format,
            source=path if plugin_format is PluginFormat.SOURCE else None,
            wheel=path if plugin_format is PluginFormat.WHEEL else None,
            package=package,
        )

    def to_config(self) -> dict[str, str]:
        """
        Serialize this spec to the explicit YAML-style mapping.

        Returns:
            dict[str, str]: The normalized configuration mapping.
        """
        config = {
            "name": self.name or "",
            "format": self.format.value if self.format else "",
            self.format.value if self.format else "source": str(self.artifact_path),
        }
        if self.package:
            config["package"] = self.package
        return config
