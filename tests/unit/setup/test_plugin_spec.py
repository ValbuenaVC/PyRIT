# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

import pytest

from pyrit.setup import PluginFormat, PluginSpec


def test_source_spec_resolves_relative_path(tmp_path: Path) -> None:
    spec = PluginSpec.from_config(
        {
            "name": "operation_foobar",
            "format": "source",
            "source": "plugins/operation_foobar.py",
        },
        base_dir=tmp_path,
    )

    assert spec.name == "operation_foobar"
    assert spec.format is PluginFormat.SOURCE
    assert spec.source == (tmp_path / "plugins" / "operation_foobar.py").resolve()
    assert spec.wheel is None
    assert spec.artifact_path == spec.source


def test_wheel_spec_resolves_package_and_path(tmp_path: Path) -> None:
    spec = PluginSpec.from_config(
        {
            "name": "partner_scenarios",
            "format": "wheel",
            "wheel": "plugins/partner.whl",
            "package": "partner_scenarios.plugin",
        },
        base_dir=tmp_path,
    )

    assert spec.name == "partner_scenarios"
    assert spec.format is PluginFormat.WHEEL
    assert spec.wheel == (tmp_path / "plugins" / "partner.whl").resolve()
    assert spec.source is None
    assert spec.package == "partner_scenarios.plugin"
    assert spec.artifact_path == spec.wheel


@pytest.mark.parametrize(
    "entry",
    [
        {},
        {"name": "x", "format": "source"},
        {"name": "x", "format": "wheel"},
        {"name": "x", "format": "source", "source": "x.py", "wheel": "x.whl"},
        {"name": "x", "format": "source", "wheel": "x.whl"},
        {"name": "x", "format": "wheel", "source": "x.py"},
        {"name": "Bad-Name", "format": "source", "source": "x.py"},
        {"name": "x", "format": "wheel", "wheel": "x.whl", "package": "bad-package"},
        {"name": "x", "format": "archive", "wheel": "x.whl"},
        {"name": "x", "format": "source", "source": "x.py", "unexpected": True},
    ],
)
def test_plugin_spec_rejects_invalid_config(entry: dict[str, object], tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        PluginSpec.from_config(entry, base_dir=tmp_path)


def test_plugin_spec_rejects_legacy_shorthand() -> None:
    with pytest.raises(ValueError, match="mapping"):
        PluginSpec.from_config("plugin.whl")  # type: ignore[arg-type]
