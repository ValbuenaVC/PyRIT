# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import zipfile
from pathlib import Path

import pytest

from pyrit.exceptions import PluginSourceNotFoundError, PluginWheelNotFoundError
from pyrit.setup import PluginFormat, PluginSpec
from pyrit.setup.plugin_formats import SourcePluginFormat, WheelPluginFormat


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


async def test_source_file_prepare_returns_common_artifact_without_execution(tmp_path: Path) -> None:
    source = tmp_path / "operation_foobar.py"
    marker = tmp_path / "executed.txt"
    source.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    spec = PluginSpec(name="operation_foobar", format=PluginFormat.SOURCE, source=source)

    prepared = await SourcePluginFormat().prepare_async(spec=spec)

    assert prepared.spec is spec
    assert prepared.import_root == tmp_path.resolve()
    assert prepared.entry_modules == ("operation_foobar",)
    assert prepared.owned_module_prefixes == ("operation_foobar",)
    assert prepared.artifact_fingerprint
    assert prepared.declared_pyrit_version is None
    assert not marker.exists()


async def test_source_file_prepare_rejects_missing_file(tmp_path: Path) -> None:
    spec = PluginSpec(
        name="missing",
        format=PluginFormat.SOURCE,
        source=tmp_path / "missing.py",
    )

    with pytest.raises(PluginSourceNotFoundError, match="existing"):
        await SourcePluginFormat().prepare_async(spec=spec)


async def test_source_file_prepare_rejects_non_python_file(tmp_path: Path) -> None:
    source = tmp_path / "plugin.txt"
    source.write_text("not Python", encoding="utf-8")
    spec = PluginSpec(name="plugin", format=PluginFormat.SOURCE, source=source)

    with pytest.raises(PluginSourceNotFoundError, match="Python"):
        await SourcePluginFormat().prepare_async(spec=spec)


async def test_source_file_fingerprint_changes_with_content(tmp_path: Path) -> None:
    source = tmp_path / "plugin.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    spec = PluginSpec(name="plugin", format=PluginFormat.SOURCE, source=source)
    adapter = SourcePluginFormat()

    first = await adapter.prepare_async(spec=spec)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = await adapter.prepare_async(spec=spec)

    assert first.artifact_fingerprint != second.artifact_fingerprint


async def test_source_package_prepare_returns_package_ownership(tmp_path: Path) -> None:
    package = tmp_path / "operation_plugin"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "scenarios.py").write_text("VALUE = 1\n", encoding="utf-8")
    spec = PluginSpec(
        name="operation",
        format=PluginFormat.SOURCE,
        source=package,
    )

    prepared = await SourcePluginFormat().prepare_async(spec=spec)

    assert prepared.import_root == tmp_path.resolve()
    assert prepared.entry_modules == ("operation_plugin",)
    assert prepared.owned_module_prefixes == ("operation_plugin",)


async def test_source_package_accepts_explicit_nested_entry_module(tmp_path: Path) -> None:
    package = tmp_path / "operation_plugin"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "entry.py").write_text("", encoding="utf-8")
    spec = PluginSpec(
        name="operation",
        format=PluginFormat.SOURCE,
        source=package,
        package="operation_plugin.entry",
    )

    prepared = await SourcePluginFormat().prepare_async(spec=spec)

    assert prepared.entry_modules == ("operation_plugin.entry",)
    assert prepared.owned_module_prefixes == ("operation_plugin",)


async def test_source_package_rejects_loose_directory(tmp_path: Path) -> None:
    source = tmp_path / "loose"
    source.mkdir()
    (source / "scenario.py").write_text("", encoding="utf-8")
    spec = PluginSpec(name="loose", format=PluginFormat.SOURCE, source=source)

    with pytest.raises(PluginSourceNotFoundError, match="__init__.py"):
        await SourcePluginFormat().prepare_async(spec=spec)


async def test_source_package_rejects_foreign_entry_module(tmp_path: Path) -> None:
    package = tmp_path / "operation_plugin"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    spec = PluginSpec(
        name="operation",
        format=PluginFormat.SOURCE,
        source=package,
        package="other_plugin",
    )

    with pytest.raises(ValueError, match="inside"):
        await SourcePluginFormat().prepare_async(spec=spec)


async def test_source_package_fingerprint_changes_with_nested_content(tmp_path: Path) -> None:
    package = tmp_path / "operation_plugin"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    nested = package / "scenario.py"
    nested.write_text("VALUE = 1\n", encoding="utf-8")
    spec = PluginSpec(name="operation", format=PluginFormat.SOURCE, source=package)
    adapter = SourcePluginFormat()

    first = await adapter.prepare_async(spec=spec)
    nested.write_text("VALUE = 2\n", encoding="utf-8")
    second = await adapter.prepare_async(spec=spec)

    assert first.artifact_fingerprint != second.artifact_fingerprint
