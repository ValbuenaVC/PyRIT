# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
TargetToModelAdapter — wraps a PyRIT PromptTarget as an Inspect AI model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrit.prompt_target import PromptTarget


class TargetToModelAdapter:
    """
    Wraps a PyRIT ``PromptTarget`` as an Inspect AI ``ModelAPI``.

    Inspect AI constructs model providers positionally; this adapter matches
    Inspect's ``ModelAPI`` positional signature and calls ``super().__init__``.

    When ``target`` is ``None``, the target is resolved from ``TargetRegistry``
    using the portion of ``model_name`` after ``"pyrit/"``. When ``target`` is
    provided directly, it is used as-is (useful for testing).

    Non-empty ``tools`` in ``generate`` raises ``InspectBridgeError`` because
    PyRIT targets are non-tool-using as Inspect models.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: Any = None,
        *,
        target: PromptTarget | None = None,
        **model_args: Any,
    ) -> None:
        """
        Initialize the TargetToModelAdapter.

        Args:
            model_name (str): Inspect model name in the form ``"pyrit/<registry_name>"``.
            base_url (str | None): Unused; accepted for Inspect ModelAPI compatibility.
            api_key (str | None): Unused; accepted for Inspect ModelAPI compatibility.
            api_key_vars (list[str] | None): Unused; accepted for Inspect ModelAPI compatibility.
            config: Optional Inspect ``GenerateConfig`` object.
            target (PromptTarget | None): PyRIT target instance. When ``None``, the
                target is resolved from ``TargetRegistry`` by the name after ``"pyrit/"``
                in ``model_name``.
            **model_args: Additional keyword arguments forwarded to ``ModelAPI.__init__``.

        """
        raise NotImplementedError

    @property
    def target(self) -> PromptTarget:
        """
        Return the wrapped PyRIT target.

        Returns:
            PromptTarget: The underlying target instance.

        """
        raise NotImplementedError

    async def generate(self, input: Any, tools: Any, tool_choice: Any, config: Any) -> Any:  # pyrit-async-suffix-exempt
        """
        Convert Inspect input to a PyRIT conversation and call the wrapped target.

        Converts the Inspect ``input`` (``list[ChatMessage]``) into a normalized
        PyRIT conversation (``list[Message]``) and calls the wrapped target's
        ``_send_prompt_to_target_async(normalized_conversation=...)`` directly
        (not the public ``send_prompt_async``, which rebuilds history from memory
        and accepts a single message). The response is then converted to a
        ``ModelOutput``.

        Args:
            input: Inspect chat messages (``list[ChatMessage]``).
            tools: Inspect tool definitions. Non-empty raises ``InspectBridgeError``.
            tool_choice: Inspect tool-choice policy (ignored; tools not supported).
            config: Inspect ``GenerateConfig`` (ignored).

        Returns:
            ModelOutput: The Inspect model output wrapping the target's response.

        Raises:
            InspectBridgeError: If ``tools`` is non-empty (PyRIT targets are
                non-tool-using as Inspect models).

        """
        raise NotImplementedError

    @staticmethod
    def model_name_for(*, target: PromptTarget) -> str:
        """
        Return the Inspect model name for a given PyRIT target.

        Args:
            target (PromptTarget): The PyRIT target instance.

        Returns:
            str: The Inspect model name in the form ``"pyrit/<unique_name>"``.

        """
        raise NotImplementedError
