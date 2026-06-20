# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
MemoryAdapter — Inspect AI Hooks subclass that persists transcripts to PyRIT CentralMemory.
"""

from __future__ import annotations

import copy
import hashlib

from pyrit.inspect_bridge._imports import require_inspect_ai
from pyrit.inspect_bridge.conversion import to_pyrit_message_pieces

# Lazily-built class; populated on first call to _get_memory_adapter_class().
_memory_adapter_class: type | None = None


def _get_memory_adapter_class() -> type:
    """
    Build (or return cached) the real MemoryAdapter class.

    The class is created at runtime so Hooks inheritance is deferred until
    inspect_ai is actually installed and requested.

    Returns:
        The ``MemoryAdapter`` class (a ``Hooks`` subclass).

    """
    global _memory_adapter_class
    if _memory_adapter_class is not None:
        return _memory_adapter_class

    require_inspect_ai()
    from inspect_ai.hooks import Hooks

    class MemoryAdapter(Hooks):
        """
        Inspect AI ``Hooks`` subclass that persists Inspect transcripts to PyRIT ``CentralMemory``.

        Inspect event/sample data is FRAMEWORK-OWNED: this adapter always deep-copies
        data before any transform and never mutates the original objects.

        The single persistence point is ``on_sample_end`` to avoid double-writes.
        ``on_run_start`` and ``on_task_start`` only capture IDs for conversation-ID mapping.
        """

        def __init__(self) -> None:
            super().__init__()
            self._current_run_id: str | None = None
            self._current_eval_id: str | None = None

        async def on_run_start(self, data: object) -> None:  # pyrit-async-suffix-exempt
            """
            Handle run-start event.

            Captures the run ID for later conversation-ID mapping.

            Args:
                data: Inspect ``RunStart`` data object.

            """
            self._current_run_id = getattr(data, "run_id", None)

        async def on_task_start(self, data: object) -> None:  # pyrit-async-suffix-exempt
            """
            Handle task-start event.

            Captures the task ID for later conversation-ID mapping.

            Args:
                data: Inspect ``TaskStart`` data object.

            """
            self._current_eval_id = getattr(data, "eval_id", None)

        async def on_sample_end(self, data: object) -> None:  # pyrit-async-suffix-exempt
            """
            Handle sample-end event (the single persistence point).

            Converts the ``EvalSample`` messages and scores to PyRIT ``MessagePiece``
            objects and persists them to ``CentralMemory``. The input ``data`` object
            is never mutated (deep-copied before transform).

            Args:
                data: Inspect ``SampleEnd`` data object. ``data.sample`` is the
                    ``EvalSample`` containing messages and scores.

            """
            from pyrit.memory.central_memory import CentralMemory

            sample = getattr(data, "sample", None)
            if sample is None:
                return

            task_id = getattr(data, "eval_id", None) or self._current_eval_id or "unknown"
            sample_id = getattr(data, "sample_id", getattr(sample, "id", 0))
            epoch = getattr(sample, "epoch", 1) or 1

            conversation_id = MemoryAdapter.conversation_id_for(
                task_id=task_id,
                sample_id=sample_id,
                epoch=epoch,
            )

            # Deep-copy just the messages list so we never mutate framework-owned objects
            raw_messages = getattr(sample, "messages", None) or []
            messages = copy.deepcopy(list(raw_messages))
            if not messages:
                return

            pieces = to_pyrit_message_pieces(
                messages=messages,
                conversation_id=conversation_id,
                sequence_start=0,
            )

            memory = CentralMemory.get_memory_instance()
            memory.add_message_pieces_to_memory(message_pieces=pieces)

        @staticmethod
        def conversation_id_for(*, task_id: str, sample_id: str | int, epoch: int = 1) -> str:
            """
            Generate a deterministic conversation ID for a given task/sample/epoch.

            The ID is a hex digest of the concatenated key components, providing
            a stable string suitable for use as a conversation identifier.

            Args:
                task_id (str): The Inspect task (eval) ID.
                sample_id (str | int): The sample ID within the task.
                epoch (int): The epoch number. Defaults to 1.

            Returns:
                str: A deterministic conversation ID string.

            """
            key = f"inspect:{task_id}:{sample_id}:{epoch}"
            return hashlib.sha256(key.encode()).hexdigest()[:32]

    _memory_adapter_class = MemoryAdapter
    return _memory_adapter_class


def __getattr__(name: str) -> object:
    if name == "MemoryAdapter":
        return _get_memory_adapter_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
