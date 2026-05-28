# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Inline tool-call parser for open chat-tuned models."""

from __future__ import annotations

import enum
import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrit.models import Message
    from pyrit.tools.models import ToolCall

logger = logging.getLogger(__name__)


class InlineToolCallParserMode(enum.Enum):
    """Policy for handling text that surrounds inline tool-call markers."""

    TRUNCATE_AT_LAST = "truncate_at_last"
    TRUNCATE_AT_FIRST = "truncate_at_first"
    EXTRACT_ALL = "extract_all"
    STRICT_TRAILING_EMPTY = "strict_trailing_empty"


class InlineToolCallParser:
    """
    Extract canonical ``ToolCall`` instances from marker-delimited JSON blocks.

    Open chat-tuned models that do not expose a structured ``tool_calls``
    channel typically emit tool calls as inline text wrapped in a
    chat-template-specific marker (an angle-bracket pair, a pipe-delimited
    tag pair, a square-bracketed list payload, and so on). This parser
    walks every ``MessagePiece`` whose ``original_value_data_type`` is
    ``"text"`` and runs ``marker_pattern`` against the piece's
    ``original_value``. Each match capture group is decoded as JSON of the
    form ``{"name": ..., "arguments": {...}}``. Synthetic ``call_id``
    values are minted positionally because inline-marker formats do not
    issue provider IDs.

    The ``mode`` parameter controls how text surrounding the markers is
    treated -- see ``InlineToolCallParserMode``. The default
    ``TRUNCATE_AT_LAST`` honors every marker but discards anything after
    the last one so hallucinated "tool results" that the model dreams up
    after the call are not persisted as if they were real outputs.

    Args:
        marker_pattern (str): Regex with exactly one capture group returning
            the JSON payload. Default targets the angle-bracket
            ``<tool_call>...</tool_call>`` syntax used by many tool-trained
            ChatML-style chat templates.
        call_id_prefix (str): Prefix for synthetic ``call_id`` values.
        mode (InlineToolCallParserMode): Surrounding-text policy.
    """

    def __init__(
        self,
        *,
        marker_pattern: str = r"<tool_call>(.*?)</tool_call>",
        call_id_prefix: str = "call",
        mode: InlineToolCallParserMode = InlineToolCallParserMode.TRUNCATE_AT_LAST,
    ) -> None:
        """
        Build an ``InlineToolCallParser``.

        Args:
            marker_pattern (str): See class docstring.
            call_id_prefix (str): See class docstring.
            mode (InlineToolCallParserMode): See class docstring.
        """
        self._pattern = re.compile(marker_pattern, re.DOTALL)
        self._call_id_prefix = call_id_prefix
        self._mode = mode

    @property
    def mode(self) -> InlineToolCallParserMode:
        """The active surrounding-text policy."""
        return self._mode

    def parse(self, message: Message) -> list[ToolCall]:
        """
        Extract tool calls from every text piece in ``message``.

        Args:
            message (Message): The most recent assistant response.

        Returns:
            list[ToolCall]: One ``ToolCall`` per valid marker match, in
                declaration order across pieces. Empty when no markers
                are found.

        Raises:
            ValueError: When ``mode`` is ``STRICT_TRAILING_EMPTY`` and any
                non-whitespace text follows the last marker in any piece.
        """
        calls: list[ToolCall] = []
        next_id = 0
        for piece in message.message_pieces:
            if piece.original_value_data_type != "text":
                continue
            matches = self._match_piece(text=piece.original_value)
            if not matches:
                continue
            for match in matches:
                call = self._build_call(match=match, next_id=next_id)
                if call is None:
                    continue
                calls.append(call)
                next_id += 1

        return calls

    def _match_piece(self, *, text: str) -> list[re.Match[str]]:
        """
        Apply the mode-specific filter to all marker matches in ``text``.

        Args:
            text (str): Piece text to scan for markers.

        Returns:
            list[re.Match[str]]: Matches to honor, in declaration order.

        Raises:
            ValueError: When ``mode`` is ``STRICT_TRAILING_EMPTY`` and any
                non-whitespace text follows the last marker.
        """
        matches = list(self._pattern.finditer(text))
        if not matches:
            return matches

        if self._mode is InlineToolCallParserMode.TRUNCATE_AT_FIRST:
            return matches[:1]
        if self._mode is InlineToolCallParserMode.STRICT_TRAILING_EMPTY:
            trailing = text[matches[-1].end() :]
            if trailing.strip():
                raise ValueError(
                    "Non-whitespace text follows the last tool-call marker; "
                    "InlineToolCallParserMode.STRICT_TRAILING_EMPTY rejects this. "
                    f"Trailing: {trailing!r}"
                )
        return matches

    def _build_call(self, *, match: re.Match[str], next_id: int) -> ToolCall | None:
        """
        Decode a single marker match into a ``ToolCall``.

        Args:
            match (re.Match[str]): A single marker match whose group 1 is the
                JSON payload.
            next_id (int): Positional id used to form ``call_id``.

        Returns:
            ToolCall | None: ``None`` when the payload is malformed or
                missing the ``name`` field. The caller is expected to log
                and skip.
        """
        from pyrit.tools.models import ToolCall

        payload = match.group(1).strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed tool-call payload: %r", payload[:120])
            return None
        if not isinstance(parsed, dict) or "name" not in parsed:
            logger.warning("Skipping tool-call payload without 'name' field: %r", payload[:120])
            return None
        arguments = self._coerce_arguments(parsed.get("arguments", {}))
        return ToolCall(
            call_id=f"{self._call_id_prefix}_{next_id}",
            name=parsed["name"],
            arguments=arguments,
            raw_envelope=parsed,
        )

    @staticmethod
    def _coerce_arguments(raw: object) -> dict:
        """
        Coerce the ``arguments`` field into a dict regardless of source shape.

        Args:
            raw (object): The value of the payload's ``arguments`` field.
                Either a dict, a JSON-encoded string, or something else.

        Returns:
            dict: The decoded arguments dict, or an empty dict on any
                shape PyRIT cannot interpret. Empty-dict fallback preserves
                the loop's behavior of "always continue with a real call".
        """
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return dict(decoded) if isinstance(decoded, dict) else {}
        return {}
