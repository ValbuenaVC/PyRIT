# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Unit tests for derived modality compatibility.

``project_request_chain`` walks a request-converter chain and reports which
``PromptDataType`` values can reach the target, or which converter broke the chain. It is
extracted from ``AttackTechniqueFactory.can_append_request_converter`` so the factory's
append check and scenario-level plan-time validation share one implementation rather than
drifting apart.
"""

from __future__ import annotations

from pyrit.converter import (
    AudioEchoConverter,
    Base64Converter,
    ImageCompressionConverter,
    QRCodeConverter,
)
from pyrit.prompt_normalizer import ConverterConfiguration
from pyrit.scenario.core.modality_validation import project_request_chain


def _configs(*converters) -> list[ConverterConfiguration]:
    """One configuration per converter, mirroring ``ConverterConfiguration.from_converters``."""
    return ConverterConfiguration.from_converters(converters=list(converters))


# ---------------------------------------------------------------------------
# Straight-line chains
# ---------------------------------------------------------------------------
def test_project_request_chain_no_converters_returns_start_types():
    """An empty chain passes the start types through untouched."""
    projected, reason = project_request_chain(start_types={"text"}, request_converters=[])
    assert projected == {"text"}
    assert reason is None


def test_project_request_chain_text_converter_preserves_text():
    """A text-to-text converter leaves the projected type as text."""
    projected, reason = project_request_chain(start_types={"text"}, request_converters=_configs(Base64Converter()))
    assert projected == {"text"}
    assert reason is None


def test_project_request_chain_image_converter_yields_image_path():
    """A text-to-image converter shifts the projected type to image_path."""
    projected, reason = project_request_chain(start_types={"text"}, request_converters=_configs(QRCodeConverter()))
    assert projected == {"image_path"}
    assert reason is None


def test_project_request_chain_audio_converter_preserves_audio():
    """An audio-to-audio converter accepts an audio start and keeps it."""
    projected, reason = project_request_chain(
        start_types={"audio_path"}, request_converters=_configs(AudioEchoConverter())
    )
    assert projected == {"audio_path"}
    assert reason is None


def test_project_request_chain_two_converters_chain():
    """Consecutive converters compose: text becomes an image, which the next converter accepts."""
    projected, reason = project_request_chain(
        start_types={"text"},
        request_converters=_configs(QRCodeConverter(), ImageCompressionConverter()),
    )
    assert projected == {"image_path"}
    assert reason is None


# ---------------------------------------------------------------------------
# Broken chains name the offending converter
# ---------------------------------------------------------------------------
def test_project_request_chain_unsupported_input_names_converter():
    """A converter that cannot accept the current type breaks the chain and is named in the reason."""
    projected, reason = project_request_chain(
        start_types={"image_path"}, request_converters=_configs(Base64Converter())
    )
    assert projected == set()
    assert reason is not None
    assert "Base64Converter" in reason
    assert "image_path" in reason


def test_project_request_chain_reports_first_failure_only():
    """The first converter that cannot accept the current type explains the failure."""
    _, reason = project_request_chain(
        start_types={"text"},
        request_converters=_configs(QRCodeConverter(), Base64Converter()),
    )
    assert reason is not None
    assert "Base64Converter" in reason
    assert "QRCodeConverter" not in reason


def test_project_request_chain_empty_start_returns_empty():
    """No start types means nothing can reach the target."""
    projected, reason = project_request_chain(start_types=set(), request_converters=_configs(Base64Converter()))
    assert projected == set()
    assert reason is None


# ---------------------------------------------------------------------------
# Conditional application
# ---------------------------------------------------------------------------
def test_project_request_chain_conditional_not_applying_preserves_type():
    """When ``prompt_data_types_to_apply`` excludes the current type the piece passes through unchanged."""
    config = ConverterConfiguration(converters=[QRCodeConverter()], prompt_data_types_to_apply=["image_path"])
    projected, reason = project_request_chain(start_types={"text"}, request_converters=[config])
    assert projected == {"text"}
    assert reason is None


def test_project_request_chain_conditional_applying_converts():
    """When ``prompt_data_types_to_apply`` includes the current type the converter runs."""
    config = ConverterConfiguration(converters=[QRCodeConverter()], prompt_data_types_to_apply=["text"])
    projected, reason = project_request_chain(start_types={"text"}, request_converters=[config])
    assert projected == {"image_path"}
    assert reason is None


def test_project_request_chain_indexes_to_apply_keeps_both_branches():
    """
    Partial application branches the projection.

    With ``indexes_to_apply`` set only some pieces are converted, so both the converted and the
    unconverted type can reach the target and the target must accept both.
    """
    config = ConverterConfiguration(converters=[QRCodeConverter()], indexes_to_apply=[0])
    projected, reason = project_request_chain(start_types={"text"}, request_converters=[config])
    assert projected == {"text", "image_path"}
    assert reason is None


def test_project_request_chain_multi_type_start_projects_each_independently():
    """Each start type is projected on its own; a conditional converter touches only what it applies to."""
    config = ConverterConfiguration(converters=[QRCodeConverter()], prompt_data_types_to_apply=["text"])
    projected, reason = project_request_chain(
        start_types={"text", "audio_path"},
        request_converters=[config],
    )
    assert projected == {"image_path", "audio_path"}
    assert reason is None


def test_project_request_chain_multi_type_start_fails_when_one_branch_breaks():
    """A converter that applies to a type it cannot accept breaks the whole chain."""
    projected, reason = project_request_chain(
        start_types={"text", "image_path"},
        request_converters=_configs(Base64Converter()),
    )
    assert projected == set()
    assert reason is not None
    assert "Base64Converter" in reason


def test_project_request_chain_image_converter_accepts_image_start():
    """A converter declaring image_path input accepts an image start with no bridge needed."""
    projected, reason = project_request_chain(
        start_types={"image_path"}, request_converters=_configs(ImageCompressionConverter())
    )
    assert projected == {"image_path"}
    assert reason is None
