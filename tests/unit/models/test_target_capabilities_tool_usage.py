# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for TargetCapabilities tool-usage fields (S3.3)."""

import pytest

from pyrit.models.target_capabilities import CapabilityName, TargetCapabilities, ToolUsageSchema

# ---------------------------------------------------------------------------
# ToolUsageSchema tests
# ---------------------------------------------------------------------------


def test_tool_usage_schema_defaults() -> None:
    schema = ToolUsageSchema()
    assert "function_call" in schema.tool_call_data_types
    assert schema.tool_result_data_type == "function_call_output"
    assert schema.tool_result_role == "tool"


def test_tool_usage_schema_custom_values() -> None:
    schema = ToolUsageSchema(
        tool_call_data_types=frozenset({"custom_call"}),
        tool_result_data_type="custom_output",
        tool_result_role="assistant",
    )
    assert schema.tool_call_data_types == frozenset({"custom_call"})
    assert schema.tool_result_data_type == "custom_output"
    assert schema.tool_result_role == "assistant"


def test_tool_usage_schema_round_trips_json() -> None:
    schema = ToolUsageSchema()
    dumped = schema.model_dump()
    restored = ToolUsageSchema(**dumped)
    assert restored.tool_call_data_types == schema.tool_call_data_types
    assert restored.tool_result_data_type == schema.tool_result_data_type
    assert restored.tool_result_role == schema.tool_result_role


# ---------------------------------------------------------------------------
# CapabilityName enum
# ---------------------------------------------------------------------------


def test_capability_name_tool_usage_enum_value() -> None:
    assert CapabilityName.TOOL_USAGE == "supports_tool_usage"


def test_capability_name_tool_usage_points_at_bool_field() -> None:
    """TOOL_USAGE must point at the bool field so includes() works."""
    caps = TargetCapabilities(supports_tool_usage=True)
    assert isinstance(getattr(caps, CapabilityName.TOOL_USAGE.value), bool)


# ---------------------------------------------------------------------------
# TargetCapabilities defaults
# ---------------------------------------------------------------------------


def test_target_capabilities_default_tool_usage_is_false() -> None:
    caps = TargetCapabilities()
    assert caps.supports_tool_usage is False


def test_target_capabilities_default_tool_usage_schema_is_none() -> None:
    caps = TargetCapabilities()
    assert caps.tool_usage_schema is None


# ---------------------------------------------------------------------------
# includes() reflects bool
# ---------------------------------------------------------------------------


def test_includes_tool_usage_false_by_default() -> None:
    caps = TargetCapabilities()
    assert caps.includes(capability=CapabilityName.TOOL_USAGE) is False


def test_includes_tool_usage_true_when_enabled() -> None:
    caps = TargetCapabilities(supports_tool_usage=True)
    assert caps.includes(capability=CapabilityName.TOOL_USAGE) is True


def test_includes_not_affected_by_schema_alone() -> None:
    """Schema being set should not affect includes() — the BOOL field governs."""
    caps = TargetCapabilities(supports_tool_usage=False, tool_usage_schema=ToolUsageSchema())
    assert caps.includes(capability=CapabilityName.TOOL_USAGE) is False


# ---------------------------------------------------------------------------
# Frozen-model immutability
# ---------------------------------------------------------------------------


def test_target_capabilities_is_frozen() -> None:
    caps = TargetCapabilities()
    with pytest.raises(Exception):  # ValidationError or TypeError
        caps.supports_tool_usage = True  # type: ignore[misc]


def test_target_capabilities_with_schema_is_frozen() -> None:
    caps = TargetCapabilities(supports_tool_usage=True, tool_usage_schema=ToolUsageSchema())
    with pytest.raises(Exception):
        caps.tool_usage_schema = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Existing capabilities unaffected
# ---------------------------------------------------------------------------


def test_existing_capability_multi_turn_unaffected() -> None:
    caps = TargetCapabilities(supports_multi_turn=True)
    assert caps.includes(capability=CapabilityName.MULTI_TURN) is True
    assert caps.supports_tool_usage is False


def test_existing_capability_system_prompt_unaffected() -> None:
    caps = TargetCapabilities(supports_system_prompt=True)
    assert caps.includes(capability=CapabilityName.SYSTEM_PROMPT) is True
    assert caps.supports_tool_usage is False


def test_all_existing_capabilities_default_false() -> None:
    caps = TargetCapabilities()
    for cap in [
        CapabilityName.MULTI_TURN,
        CapabilityName.MULTI_MESSAGE_PIECES,
        CapabilityName.JSON_SCHEMA,
        CapabilityName.JSON_OUTPUT,
        CapabilityName.EDITABLE_HISTORY,
        CapabilityName.SYSTEM_PROMPT,
        CapabilityName.STREAMING_AUDIO,
    ]:
        assert caps.includes(capability=cap) is False


# ---------------------------------------------------------------------------
# Schema alongside tool-usage bool
# ---------------------------------------------------------------------------


def test_target_capabilities_with_tool_usage_schema() -> None:
    schema = ToolUsageSchema()
    caps = TargetCapabilities(supports_tool_usage=True, tool_usage_schema=schema)
    assert caps.supports_tool_usage is True
    assert caps.tool_usage_schema is not None
    assert caps.includes(capability=CapabilityName.TOOL_USAGE) is True
