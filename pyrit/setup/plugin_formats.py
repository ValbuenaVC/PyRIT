# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Artifact-format adapters for scenario plug-in consumption."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pyrit.exceptions import PluginSourceNotFoundError, PluginWheelNotFoundError
from pyrit.setup.plugin_spec import PluginFormat, PluginSpec


@dataclass(frozen=True)
class PreparedPlugin:
    """A prepared import root shared by every plug-in artifact format."""

    spec: PluginSpec
    import_root: Path
    entry_modules: tuple[str, ...]
    owned_module_prefixes: tuple[str, ...]
    artifact_fingerprint: str
    declared_pyrit_version: str | None = None


class SourcePluginFormat:
    """Prepare a server-local Python source file for plug-in discovery."""

    async def prepare_async(self, *, spec: PluginSpec) -> PreparedPlugin:
        """
        Validate one source-file plug-in without importing or executing it.

        Args:
            spec (PluginSpec): The normalized source declaration.

        Returns:
            PreparedPlugin: The prepared source import root and ownership metadata.

        Raises:
            ValueError: If the spec is not source format.
            PluginSourceNotFoundError: If the source is absent or not a Python file.
        """
        return await asyncio.to_thread(self._prepare, spec=spec)

    def _prepare(self, *, spec: PluginSpec) -> PreparedPlugin:
        if spec.format is not PluginFormat.SOURCE or spec.source is None:
            raise ValueError("SourcePluginFormat requires a source-format PluginSpec.")

        source = spec.source.expanduser().resolve()
        if not source.is_file():
            raise PluginSourceNotFoundError(f"Plug-in source does not point to an existing Python file: {source}")
        if source.suffix != ".py" or not source.stem.isidentifier():
            raise PluginSourceNotFoundError(
                f"Plug-in source must be a Python file with an importable filename: {source}"
            )

        return PreparedPlugin(
            spec=spec,
            import_root=source.parent,
            entry_modules=(source.stem,),
            owned_module_prefixes=(source.stem,),
            artifact_fingerprint=self._fingerprint(source=source),
        )

    @staticmethod
    def _fingerprint(*, source: Path) -> str:
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


class WheelPluginFormat:
    """Prepare a pre-built Python wheel for plug-in discovery."""

    _FINGERPRINT_FILE = ".plugin_wheel_fingerprint"

    def __init__(self, *, base_dir: Path | None = None) -> None:
        """
        Initialize the wheel adapter.

        Args:
            base_dir (Path | None): Extraction root. Defaults to ``PLUGIN_DIR`` or
                PyRIT's ``.plugin`` directory.
        """
        self._base_dir = base_dir

    async def prepare_async(self, *, spec: PluginSpec) -> PreparedPlugin:
        """
        Validate and extract one wheel-format plug-in.

        Args:
            spec (PluginSpec): The normalized wheel declaration.

        Returns:
            PreparedPlugin: The prepared wheel import root and ownership metadata.

        Raises:
            ValueError: If the spec is not wheel format or package discovery is ambiguous.
            PluginWheelNotFoundError: If the wheel path is absent or invalid.
        """
        return await asyncio.to_thread(self._prepare, spec=spec)

    def _prepare(self, *, spec: PluginSpec) -> PreparedPlugin:
        if spec.format is not PluginFormat.WHEEL or spec.wheel is None:
            raise ValueError("WheelPluginFormat requires a wheel-format PluginSpec.")
        wheel = spec.wheel.expanduser()
        if not wheel.is_file():
            raise PluginWheelNotFoundError(f"Plug-in wheel does not point to an existing file: {wheel}")
        if wheel.suffix != ".whl":
            raise PluginWheelNotFoundError(f"Plug-in wheel must be a .whl file, got: {wheel}")

        fingerprint = self._fingerprint(wheel=wheel)
        extract_dir = self._extract(wheel=wheel, fingerprint=fingerprint)
        package = self._resolve_package(extract_dir=extract_dir, explicit_package=spec.package)
        return PreparedPlugin(
            spec=spec,
            import_root=extract_dir,
            entry_modules=(package,),
            owned_module_prefixes=(package,),
            artifact_fingerprint=fingerprint,
            declared_pyrit_version=self._read_required_pyrit_version(extract_dir=extract_dir),
        )

    def _extract(self, *, wheel: Path, fingerprint: str) -> Path:
        from pyrit.common.safe_extract import safe_extract_zip

        base_dir = self._resolve_base_dir()
        base_dir.mkdir(parents=True, exist_ok=True)
        extract_dir = base_dir / wheel.stem
        marker = extract_dir / self._FINGERPRINT_FILE
        if extract_dir.is_dir() and marker.is_file():
            try:
                if marker.read_text(encoding="utf-8").strip() == fingerprint:
                    return extract_dir
            except OSError:
                pass

        temp_dir = Path(tempfile.mkdtemp(prefix=f".{wheel.stem}.tmp-", dir=base_dir))
        try:
            safe_extract_zip(source=wheel, dest_dir=temp_dir)
            (temp_dir / self._FINGERPRINT_FILE).write_text(fingerprint, encoding="utf-8")
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            os.replace(temp_dir, extract_dir)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        return extract_dir

    def _resolve_base_dir(self) -> Path:
        if self._base_dir is not None:
            return self._base_dir.expanduser().resolve()
        override = os.getenv("PLUGIN_DIR")
        if override:
            return Path(override).expanduser().resolve()
        from pyrit.common import path

        return (Path(path.HOME_PATH) / ".plugin").resolve()

    @staticmethod
    def _fingerprint(*, wheel: Path) -> str:
        digest = hashlib.sha256()
        with wheel.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _resolve_package(*, extract_dir: Path, explicit_package: str | None) -> str:
        if explicit_package:
            return explicit_package
        for dist_info in sorted(extract_dir.glob("*.dist-info")):
            top_level = dist_info / "top_level.txt"
            if top_level.is_file():
                names = [line.strip() for line in top_level.read_text(encoding="utf-8").splitlines() if line.strip()]
                if len(names) == 1:
                    return names[0]

        candidates = sorted(
            child.name
            for child in extract_dir.iterdir()
            if child.is_dir() and not child.name.endswith((".dist-info", ".data")) and (child / "__init__.py").is_file()
        )
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(f"Could not find an importable top-level package in {extract_dir}.")
        raise ValueError(f"Found multiple top-level packages in {extract_dir}: {candidates}.")

    @staticmethod
    def _read_required_pyrit_version(*, extract_dir: Path) -> str | None:
        import re

        for metadata in extract_dir.glob("*.dist-info/METADATA"):
            try:
                text = metadata.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                if not re.match(r"Requires-Dist:\s*pyrit\b", line, flags=re.IGNORECASE):
                    continue
                match = re.search(r"==\s*([0-9][0-9A-Za-z.\-]*)", line)
                if match:
                    return match.group(1)
        return None
