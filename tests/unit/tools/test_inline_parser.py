# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for InlineToolCallParser across marker syntaxes."""

from __future__ import annotations

import logging
import uuid

import pytest

from pyrit.models import Message, MessagePiece
from pyrit.tools import InlineToolCallParser, InlineToolCallParserMode


def _assistant_text(text: str) -> Message:
    return Message(
        message_pieces=[
            MessagePiece(
                role="assistant",
                original_value=text,
                original_value_data_type="text",
                conversation_id=str(uuid.uuid4()),
            )
        ],
        skip_validation=True,
    )


# Marker patterns commonly seen in tool-trained open chat templates.
# Named after their syntactic shape, not the model family that uses them,
# so that PyRIT does not advertise a "supported vendors" list.
ANGLE_BRACKET_PATTERN = r"<tool_call>(.*?)</tool_call>"
PIPE_PYTHON_TAG_PATTERN = r"<\|python_tag\|>(.*?)<\|eom_id\|>"
SQUARE_BRACKET_LIST_PATTERN = r"\[TOOL_CALLS\]\s*(\[.*?\])"


# ---------------------------------------------------------------------------
# Marker-pattern coverage: angle-bracket / pipe-python-tag / square-bracket-list
# ---------------------------------------------------------------------------


class TestAngleBracketMarker:
    """Default pattern: ``<tool_call>...</tool_call>``."""

    def test_single_marker_extracts_call(self):
        parser = InlineToolCallParser()
        text = (
            'Let me check the weather. <tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>'
        )
        calls = parser.parse(_assistant_text(text))
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"location": "NYC"}
        assert calls[0].call_id == "call_0"

    def test_multiple_markers_extract_all_with_synthetic_ids(self):
        parser = InlineToolCallParser(mode=InlineToolCallParserMode.EXTRACT_ALL)
        text = (
            '<tool_call>{"name": "f1", "arguments": {}}</tool_call>'
            "some interleaving text "
            '<tool_call>{"name": "f2", "arguments": {"k": 1}}</tool_call>'
        )
        calls = parser.parse(_assistant_text(text))
        assert [c.name for c in calls] == ["f1", "f2"]
        assert [c.call_id for c in calls] == ["call_0", "call_1"]

    def test_arguments_as_json_string_is_decoded(self):
        parser = InlineToolCallParser()
        # arguments doubly-encoded as a JSON string (canonical envelope shape).
        text = '<tool_call>{"name": "f", "arguments": "{\\"a\\": 1}"}</tool_call>'
        calls = parser.parse(_assistant_text(text))
        assert len(calls) == 1
        assert calls[0].arguments == {"a": 1}


class TestPipePythonTagMarker:
    """Pipe-delimited tag pair: ``<|python_tag|>...<|eom_id|>``."""

    def test_single_marker_extracts_call(self):
        parser = InlineToolCallParser(marker_pattern=PIPE_PYTHON_TAG_PATTERN)
        text = (
            'Sure, calling now. <|python_tag|>{"name": "get_weather", "arguments": {"location": "Seattle"}}<|eom_id|>'
        )
        calls = parser.parse(_assistant_text(text))
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"location": "Seattle"}

    def test_multi_marker_extract_all(self):
        parser = InlineToolCallParser(
            marker_pattern=PIPE_PYTHON_TAG_PATTERN,
            mode=InlineToolCallParserMode.EXTRACT_ALL,
        )
        text = (
            '<|python_tag|>{"name": "a", "arguments": {}}<|eom_id|>'
            "between "
            '<|python_tag|>{"name": "b", "arguments": {}}<|eom_id|>'
        )
        calls = parser.parse(_assistant_text(text))
        assert [c.name for c in calls] == ["a", "b"]


class TestSquareBracketListMarker:
    """Square-bracketed list payload: ``[TOOL_CALLS] [...]``."""

    def test_list_payload_is_skipped_with_warning(self, caplog):
        """The default parser expects a single-dict payload.

        Marker syntaxes whose payload is a JSON LIST of dicts (rather than a
        single dict) are logged and dropped. Callers that need list-shaped
        payloads should either subclass ``InlineToolCallParser`` and override
        ``parse`` to iterate the list, or use a marker pattern that targets
        each dict inside the list separately. Regex does not handle nested
        braces well; subclassing is cleaner.
        """
        parser = InlineToolCallParser(marker_pattern=SQUARE_BRACKET_LIST_PATTERN)
        text = '[TOOL_CALLS] [{"name": "f", "arguments": {"x": 1}}]'
        with caplog.at_level(logging.WARNING, logger="pyrit.tools.inline_parser"):
            calls = parser.parse(_assistant_text(text))
        assert calls == []
        assert any("without 'name' field" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Mode coverage: TRUNCATE_AT_LAST / TRUNCATE_AT_FIRST / EXTRACT_ALL /
#                STRICT_TRAILING_EMPTY
# ---------------------------------------------------------------------------


class TestParserModes:
    """Surrounding-text policy coverage."""

    HALLUCINATED = (
        '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>'
        " The weather in NYC is sunny and 72 degrees."
    )
    DOUBLE_CALL = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        " middle "
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
        " trailing chatter"
    )

    def test_truncate_at_last_default_drops_trailing_chatter(self):
        parser = InlineToolCallParser()  # default mode
        calls = parser.parse(_assistant_text(self.HALLUCINATED))
        # The hallucinated weather report after the marker is discarded; the
        # call itself is honored.
        assert len(calls) == 1
        assert calls[0].name == "get_weather"

    def test_truncate_at_last_extracts_all_markers_then_drops_tail(self):
        parser = InlineToolCallParser(mode=InlineToolCallParserMode.TRUNCATE_AT_LAST)
        calls = parser.parse(_assistant_text(self.DOUBLE_CALL))
        # Both markers honored, trailing "trailing chatter" silently dropped.
        assert [c.name for c in calls] == ["a", "b"]

    def test_truncate_at_first_keeps_only_the_first_marker(self):
        parser = InlineToolCallParser(mode=InlineToolCallParserMode.TRUNCATE_AT_FIRST)
        calls = parser.parse(_assistant_text(self.DOUBLE_CALL))
        assert [c.name for c in calls] == ["a"]

    def test_extract_all_keeps_every_marker(self):
        parser = InlineToolCallParser(mode=InlineToolCallParserMode.EXTRACT_ALL)
        calls = parser.parse(_assistant_text(self.DOUBLE_CALL))
        assert [c.name for c in calls] == ["a", "b"]

    def test_strict_trailing_empty_raises_on_chatter(self):
        parser = InlineToolCallParser(mode=InlineToolCallParserMode.STRICT_TRAILING_EMPTY)
        with pytest.raises(ValueError, match="STRICT_TRAILING_EMPTY"):
            parser.parse(_assistant_text(self.HALLUCINATED))

    def test_strict_trailing_empty_passes_when_only_whitespace_after(self):
        parser = InlineToolCallParser(mode=InlineToolCallParserMode.STRICT_TRAILING_EMPTY)
        text = '<tool_call>{"name": "f", "arguments": {}}</tool_call>\n  \t  '
        calls = parser.parse(_assistant_text(text))
        assert [c.name for c in calls] == ["f"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Empty input, malformed JSON, missing name, multi-piece messages."""

    def test_no_markers_returns_empty(self):
        parser = InlineToolCallParser()
        calls = parser.parse(_assistant_text("just plain assistant text"))
        assert calls == []

    def test_malformed_json_is_skipped_silently(self):
        parser = InlineToolCallParser()
        text = "<tool_call>not valid json</tool_call>"
        calls = parser.parse(_assistant_text(text))
        assert calls == []

    def test_payload_without_name_is_skipped(self):
        parser = InlineToolCallParser()
        text = '<tool_call>{"arguments": {}}</tool_call>'
        calls = parser.parse(_assistant_text(text))
        assert calls == []

    def test_non_text_pieces_are_ignored(self):
        """Pieces with data_type other than 'text' are skipped entirely."""
        parser = InlineToolCallParser()
        msg = Message(
            message_pieces=[
                MessagePiece(
                    role="assistant",
                    original_value='<tool_call>{"name": "ignored", "arguments": {}}</tool_call>',
                    original_value_data_type="reasoning",
                    conversation_id=str(uuid.uuid4()),
                ),
                MessagePiece(
                    role="assistant",
                    original_value='<tool_call>{"name": "found", "arguments": {}}</tool_call>',
                    original_value_data_type="text",
                    conversation_id=str(uuid.uuid4()),
                ),
            ],
            skip_validation=True,
        )
        calls = parser.parse(msg)
        assert [c.name for c in calls] == ["found"]

    def test_call_id_prefix_customization(self):
        parser = InlineToolCallParser(call_id_prefix="custom")
        text = '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
        calls = parser.parse(_assistant_text(text))
        assert calls[0].call_id == "custom_0"
