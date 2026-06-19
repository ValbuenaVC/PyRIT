# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
MemoryAdapter — Inspect AI Hooks subclass that persists transcripts to PyRIT CentralMemory.
"""

from __future__ import annotations

from typing import Any


class MemoryAdapter:
    """
    Inspect AI ``Hooks`` subclass that persists Inspect transcripts to PyRIT ``CentralMemory``.

    Inspect event/sample data is FRAMEWORK-OWNED: this adapter always deep-copies
    data before any transform and never mutates the original objects.

    The single persistence point is ``on_sample_end`` to avoid double-writes.
    ``on_run_start`` and ``on_task_start`` only capture IDs for conversation-ID mapping.
    """

    async def on_run_start(self, data: Any) -> None:  # type: ignore[name-defined]  # noqa: F821  # pyrit-async-suffix-exempt
        """
        Handle run-start event.

        Captures the run ID for later conversation-ID mapping.

        Args:
            data: Inspect ``RunStart`` data object.

        """
        raise NotImplementedError

    async def on_task_start(self, data: Any) -> None:  # type: ignore[name-defined]  # noqa: F821  # pyrit-async-suffix-exempt
        """
        Handle task-start event.

        Captures the task ID for later conversation-ID mapping.

        Args:
            data: Inspect ``TaskStart`` data object.

        """
        raise NotImplementedError

    async def on_sample_end(self, data: Any) -> None:  # type: ignore[name-defined]  # noqa: F821  # pyrit-async-suffix-exempt
        """
        Handle sample-end event (the single persistence point).

        Converts the ``EvalSample`` messages and scores to PyRIT ``MessagePiece``
        and ``Score`` objects and persists them to ``CentralMemory``. The input
        ``data`` object is never mutated (deep-copied before transform).

        Args:
            data: Inspect ``SampleEnd`` data object. ``data.sample`` is the
                ``EvalSample`` containing messages and scores.

        """
        raise NotImplementedError

    @staticmethod
    def conversation_id_for(*, task_id: str, sample_id: str | int, epoch: int = 1) -> str:
        """
        Generate a deterministic conversation ID for a given task/sample/epoch.

        Args:
            task_id (str): The Inspect task (eval) ID.
            sample_id (str | int): The sample ID within the task.
            epoch (int): The epoch number. Defaults to 1.

        Returns:
            str: A deterministic conversation ID string.

        """
        raise NotImplementedError
