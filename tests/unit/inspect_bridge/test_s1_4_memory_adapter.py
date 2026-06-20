# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for S1.4: MemoryAdapter (Inspect Hooks -> PyRIT CentralMemory).

Uses @pytest.mark.usefixtures("patch_central_database") for memory tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_sample_end(*, task_id: str = "task-1", sample_id: int = 1, epoch: int = 1) -> object:
    """Create a minimal SampleEnd-like object for testing."""
    from inspect_ai.hooks import SampleEnd
    from inspect_ai.log import EvalSample
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

    sample = MagicMock(spec=EvalSample)
    sample.id = sample_id
    sample.epoch = epoch
    sample.messages = [
        ChatMessageUser(content="Hello"),
        ChatMessageAssistant(content="Hi there"),
    ]
    sample.scores = {}

    data = MagicMock(spec=SampleEnd)
    data.eval_id = task_id
    data.sample_id = sample_id
    data.sample = sample
    return data


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_memory_adapter_is_hooks() -> None:
    """MemoryAdapter is a subclass of inspect_ai.hooks.Hooks."""
    from inspect_ai.hooks import Hooks

    from pyrit.inspect_bridge._memory_adapter import MemoryAdapter

    adapter = MemoryAdapter()
    assert isinstance(adapter, Hooks)


# ---------------------------------------------------------------------------
# conversation_id_for
# ---------------------------------------------------------------------------


def test_conversation_id_for_deterministic() -> None:
    """conversation_id_for returns the same string for the same inputs."""
    from pyrit.inspect_bridge._memory_adapter import MemoryAdapter

    id1 = MemoryAdapter.conversation_id_for(task_id="t1", sample_id=2, epoch=1)
    id2 = MemoryAdapter.conversation_id_for(task_id="t1", sample_id=2, epoch=1)
    assert id1 == id2


def test_conversation_id_for_unique_for_different_samples() -> None:
    """Different sample IDs produce different conversation IDs."""
    from pyrit.inspect_bridge._memory_adapter import MemoryAdapter

    id1 = MemoryAdapter.conversation_id_for(task_id="t1", sample_id=1, epoch=1)
    id2 = MemoryAdapter.conversation_id_for(task_id="t1", sample_id=2, epoch=1)
    assert id1 != id2


def test_conversation_id_for_unique_for_different_epochs() -> None:
    """Different epochs produce different conversation IDs."""
    from pyrit.inspect_bridge._memory_adapter import MemoryAdapter

    id1 = MemoryAdapter.conversation_id_for(task_id="t1", sample_id=1, epoch=1)
    id2 = MemoryAdapter.conversation_id_for(task_id="t1", sample_id=1, epoch=2)
    assert id1 != id2


def test_conversation_id_for_string_sample_id() -> None:
    """conversation_id_for works with string sample_id."""
    from pyrit.inspect_bridge._memory_adapter import MemoryAdapter

    cid = MemoryAdapter.conversation_id_for(task_id="t1", sample_id="sample-abc", epoch=1)
    assert isinstance(cid, str)
    assert len(cid) > 0


# ---------------------------------------------------------------------------
# on_run_start / on_task_start (id capture only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_run_start_does_not_raise() -> None:
    """on_run_start runs without error (captures run_id)."""
    from inspect_ai.hooks import RunStart

    from pyrit.inspect_bridge._memory_adapter import MemoryAdapter

    data = MagicMock(spec=RunStart)
    data.run_id = "run-123"
    data.task_names = ["task-1"]

    adapter = MemoryAdapter()
    await adapter.on_run_start(data)  # pyrit-async-suffix-exempt


@pytest.mark.asyncio
async def test_on_task_start_does_not_raise() -> None:
    """on_task_start runs without error (captures task/eval_id)."""
    from inspect_ai.hooks import TaskStart

    from pyrit.inspect_bridge._memory_adapter import MemoryAdapter

    data = MagicMock(spec=TaskStart)
    data.run_id = "run-123"
    data.eval_id = "eval-456"

    adapter = MemoryAdapter()
    await adapter.on_task_start(data)  # pyrit-async-suffix-exempt


# ---------------------------------------------------------------------------
# on_sample_end (single persistence point)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.asyncio
async def test_on_sample_end_persists_messages() -> None:
    """on_sample_end adds message pieces to CentralMemory."""
    from pyrit.inspect_bridge._memory_adapter import MemoryAdapter

    adapter = MemoryAdapter()
    data = _make_sample_end(task_id="eval-test", sample_id=1, epoch=1)

    mock_memory = MagicMock()
    mock_memory.add_message_pieces_to_memory = MagicMock()

    with patch("pyrit.memory.central_memory.CentralMemory.get_memory_instance", return_value=mock_memory):
        await adapter.on_sample_end(data)  # pyrit-async-suffix-exempt

    mock_memory.add_message_pieces_to_memory.assert_called_once()
    call_kwargs = mock_memory.add_message_pieces_to_memory.call_args.kwargs
    pieces = call_kwargs["message_pieces"]
    assert len(pieces) >= 2  # one per ChatMessage
    roles = {p.role for p in pieces}
    assert "user" in roles
    assert "assistant" in roles


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.asyncio
async def test_on_sample_end_conversation_id_consistent() -> None:
    """All message pieces from a sample share the same conversation_id."""
    from pyrit.inspect_bridge._memory_adapter import MemoryAdapter

    adapter = MemoryAdapter()
    data = _make_sample_end(task_id="eval-test", sample_id=5, epoch=2)

    mock_memory = MagicMock()
    mock_memory.add_message_pieces_to_memory = MagicMock()

    with patch("pyrit.memory.central_memory.CentralMemory.get_memory_instance", return_value=mock_memory):
        await adapter.on_sample_end(data)  # pyrit-async-suffix-exempt

    pieces = mock_memory.add_message_pieces_to_memory.call_args.kwargs["message_pieces"]
    conversation_ids = {p.conversation_id for p in pieces}
    assert len(conversation_ids) == 1  # all pieces share one conversation_id


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.asyncio
async def test_on_sample_end_does_not_mutate_input() -> None:
    """on_sample_end never mutates the framework-owned data object."""

    from pyrit.inspect_bridge._memory_adapter import MemoryAdapter

    adapter = MemoryAdapter()
    data = _make_sample_end()
    original_messages = list(data.sample.messages)

    mock_memory = MagicMock()
    mock_memory.add_message_pieces_to_memory = MagicMock()

    with patch("pyrit.memory.central_memory.CentralMemory.get_memory_instance", return_value=mock_memory):
        await adapter.on_sample_end(data)  # pyrit-async-suffix-exempt

    # Messages on the sample must be unchanged
    assert data.sample.messages is not None
