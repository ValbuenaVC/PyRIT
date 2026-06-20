# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
TargetToModelAdapter -- wraps a PyRIT PromptTarget as an Inspect AI model.

The real class (which subclasses inspect_ai.model.ModelAPI) is built lazily on
first access so that importing this module never triggers an inspect_ai import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyrit.inspect_bridge._imports import require_inspect_ai
from pyrit.inspect_bridge.errors import InspectBridgeError

if TYPE_CHECKING:
    from inspect_ai.model import ModelOutput

    from pyrit.prompt_target import PromptTarget

# The provider prefix used in inspect model names (e.g. "pyrit/my_target").
_PROVIDER_PREFIX: str = "pyrit"

# Lazily-built class; populated on first call to _get_adapter_class().
_adapter_class: type | None = None


def _get_adapter_class() -> type:
    """
    Build (or return cached) the real TargetToModelAdapter class.

    The class is created at runtime so ModelAPI inheritance is deferred until
    inspect_ai is actually installed and requested.

    Returns:
        The ``TargetToModelAdapter`` class (a ``ModelAPI`` subclass).

    """
    global _adapter_class
    if _adapter_class is not None:
        return _adapter_class

    require_inspect_ai()
    from inspect_ai.model import ModelAPI

    class TargetToModelAdapter(ModelAPI):
        """
        Wraps a PyRIT `PromptTarget` as an Inspect AI `ModelAPI`.

        Inspect AI constructs model providers positionally; this adapter matches
        Inspect's `ModelAPI` positional signature and calls super().__init__.

        When `target` is `None`, the target is resolved from `TargetRegistry`
        using the portion of `model_name` after `"pyrit/"`. When `target` is
        provided directly, it is used as-is (useful for testing).

        Non-empty `tools` in `generate` raises `InspectBridgeError` because
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
            super().__init__(model_name, base_url, api_key, api_key_vars or [], config)
            if target is None:
                from pyrit.registry.object_registries.target_registry import TargetRegistry

                registry = TargetRegistry.get_registry_singleton()
                name = model_name.removeprefix(f"{_PROVIDER_PREFIX}/")
                resolved = registry.get_instance_by_name(name)
                if resolved is None:
                    raise InspectBridgeError(
                        message=(
                            f"No PyRIT target registered as '{name}'. "
                            "Register the target via InspectInitializer before running tasks."
                        )
                    )
                self._target: PromptTarget = resolved
            else:
                self._target = target

        @property
        def target(self) -> PromptTarget:
            """Return the wrapped PyRIT target."""
            return self._target

        async def generate(  # pyrit-async-suffix-exempt  # noqa: A002
            self,
            input: Any,  # noqa: A002
            tools: Any,
            tool_choice: Any,
            config: Any,
        ) -> ModelOutput:
            """
            Convert Inspect input to a PyRIT conversation and return `ModelOutput`.

            Args:
                input: Inspect chat messages (`list[ChatMessage]`).
                tools: Inspect tool definitions. Non-empty raises `InspectBridgeError`.
                tool_choice: Inspect tool-choice policy (ignored; tools not supported).
                config: Inspect `GenerateConfig` (ignored).

            Returns:
                ModelOutput wrapping the target's response.

            Raises:
                InspectBridgeError: If `tools` is non-empty.

            """
            if tools:
                raise InspectBridgeError(
                    message=(
                        "PyRIT targets exposed as Inspect models do not support tool use. "
                        "Remove tools from the task or use a native Inspect model provider."
                    )
                )

            from pyrit.inspect_bridge.conversion import to_model_output, to_pyrit_message_pieces
            from pyrit.models import Message

            pyrit_pieces = to_pyrit_message_pieces(
                messages=input,
                conversation_id="inspect-bridge",
                sequence_start=0,
            )
            pyrit_messages = [Message(message_pieces=[piece]) for piece in pyrit_pieces]

            response_messages = await self._target._send_prompt_to_target_async(
                normalized_conversation=pyrit_messages
            )
            return to_model_output(messages=response_messages)

        @staticmethod
        def model_name_for(*, target: PromptTarget) -> str:
            """
            Return the Inspect model name for a given PyRIT target.

            Args:
                target: The PyRIT target instance.

            Returns:
                str: The model name in the form `"pyrit/<unique_name>"`.

            """
            return f"{_PROVIDER_PREFIX}/{target.get_identifier().unique_name}"

    _adapter_class = TargetToModelAdapter
    return _adapter_class


def __getattr__(name: str) -> object:
    if name == "TargetToModelAdapter":
        return _get_adapter_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
