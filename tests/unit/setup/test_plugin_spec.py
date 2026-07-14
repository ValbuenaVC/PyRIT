# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the source-only, initializer-pointing ``PluginSpec`` schema."""

from pathlib import Path

import pytest

from pyrit.setup import PluginSpec


def test_from_config_normalizes_source_and_initializer(tmp_path: Path) -> None:
    entry = {
        "name": "rapid_response",
        "source": "pkg_root",
        "initializer": "pyrit_internal.setup.RapidResponseInitializer",
    }
    spec = PluginSpec.from_config(entry, base_dir=tmp_path)
    assert spec.name == "rapid_response"
    assert spec.source == (tmp_path / "pkg_root").resolve()
    assert spec.initializer == "pyrit_internal.setup.RapidResponseInitializer"


def test_from_config_keeps_absolute_source(tmp_path: Path) -> None:
    absolute = (tmp_path / "artifacts").resolve()
    spec = PluginSpec.from_config(
        {"name": "p", "source": str(absolute), "initializer": "m.C"},
        base_dir=tmp_path / "elsewhere",
    )
    assert spec.source == absolute


def test_to_config_round_trips() -> None:
    spec = PluginSpec(name="p", source=Path("/opt/plugin").resolve(), initializer="m.C")
    assert spec.to_config() == {
        "name": "p",
        "source": str(Path("/opt/plugin").resolve()),
        "initializer": "m.C",
    }


@pytest.mark.parametrize(
    "entry",
    [
        {"source": "s", "initializer": "m.C"},  # missing name
        {"name": "p", "initializer": "m.C"},  # missing source
        {"name": "p", "source": "s"},  # missing initializer
        {"name": "p", "source": "s", "initializer": "no_dot"},  # non-dotted initializer
        {"name": "p", "source": "s", "initializer": "m.C", "wheel": "x"},  # unexpected key
    ],
)
def test_from_config_rejects_malformed_entries(entry: dict) -> None:
    with pytest.raises(ValueError):
        PluginSpec.from_config(entry)
