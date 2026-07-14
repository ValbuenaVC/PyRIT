# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import zipfile
from pathlib import Path

import pytest

from pyrit.exceptions import PluginWheelNotFoundError
from pyrit.setup import PluginFormat, PluginSpec
from pyrit.setup.plugin_formats import WheelPluginFormat


def _build_wheel(tmp_path: Path, *, package: str = "sample_plugin") -> Path:
    wheel = tmp_path / f"{package}-1.0.0-py3-none-any.whl"
    dist_info = f"{package}-1.0.0.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{package}/__init__.py", "")
        archive.writestr(f"{dist_info}/top_level.txt", package)
        archive.writestr(
            f"{dist_info}/METADATA",
            "\n".join(
                [
                    "Metadata-Version: 2.1",
                    f"Name: {package}",
                    "Version: 1.0.0",
                    "Requires-Dist: pyrit==0.14.0",
                ]
            ),
        )
    return wheel


async def test_wheel_prepare_returns_common_artifact(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path)
    spec = PluginSpec(
        name="sample",
        format=PluginFormat.WHEEL,
        wheel=wheel,
        package="sample_plugin",
    )

    prepared = await WheelPluginFormat(base_dir=tmp_path / ".plugin").prepare_async(spec=spec)

    assert prepared.spec is spec
    assert prepared.import_root.is_dir()
    assert prepared.entry_modules == ("sample_plugin",)
    assert prepared.owned_module_prefixes == ("sample_plugin",)
    assert prepared.artifact_fingerprint
    assert prepared.declared_pyrit_version == "0.14.0"
    assert (prepared.import_root / "sample_plugin" / "__init__.py").is_file()


async def test_wheel_prepare_resolves_package_from_metadata(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path, package="metadata_plugin")
    spec = PluginSpec(name="metadata", format=PluginFormat.WHEEL, wheel=wheel)

    prepared = await WheelPluginFormat(base_dir=tmp_path / ".plugin").prepare_async(spec=spec)

    assert prepared.entry_modules == ("metadata_plugin",)


async def test_wheel_prepare_rejects_missing_wheel(tmp_path: Path) -> None:
    spec = PluginSpec(
        name="missing",
        format=PluginFormat.WHEEL,
        wheel=tmp_path / "missing.whl",
    )

    with pytest.raises(PluginWheelNotFoundError, match="existing"):
        await WheelPluginFormat(base_dir=tmp_path / ".plugin").prepare_async(spec=spec)


async def test_wheel_prepare_rejects_source_spec(tmp_path: Path) -> None:
    spec = PluginSpec(
        name="source",
        format=PluginFormat.SOURCE,
        source=tmp_path / "plugin.py",
    )

    with pytest.raises(ValueError, match="wheel-format"):
        await WheelPluginFormat(base_dir=tmp_path / ".plugin").prepare_async(spec=spec)


async def test_wheel_prepare_reuses_unchanged_extraction(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path)
    spec = PluginSpec(name="sample", format=PluginFormat.WHEEL, wheel=wheel)
    adapter = WheelPluginFormat(base_dir=tmp_path / ".plugin")

    first = await adapter.prepare_async(spec=spec)
    marker = first.import_root / "cache_marker.txt"
    marker.write_text("kept", encoding="utf-8")
    second = await adapter.prepare_async(spec=spec)

    assert first.import_root == second.import_root
    assert marker.is_file()


def test_wheel_package_resolution_prefers_explicit(tmp_path: Path) -> None:
    assert (
        WheelPluginFormat._resolve_package(
            extract_dir=tmp_path,
            explicit_package="explicit_pkg",
        )
        == "explicit_pkg"
    )


def test_wheel_package_resolution_rejects_ambiguous_directory(tmp_path: Path) -> None:
    for name in ("pkg_a", "pkg_b"):
        package_dir = tmp_path / name
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="multiple"):
        WheelPluginFormat._resolve_package(extract_dir=tmp_path, explicit_package=None)
