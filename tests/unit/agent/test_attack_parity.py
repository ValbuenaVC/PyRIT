# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""S3.5 — Attack interface parity: Agent accepted wherever PromptTarget is accepted."""

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.agent.agent import Agent
from pyrit.agent.runtime import InProcessRuntime
from pyrit.agent.tools import Tool
from pyrit.executor.attack import (
    AttackParameters,
    PromptSendingAttack,
    SingleTurnAttackContext,
)
from pyrit.models import Message, MessagePiece
from pyrit.models.target_capabilities import TargetCapabilities, ToolUsageSchema
from pyrit.prompt_target import PromptTarget
from unit.mocks import MockPromptTarget, get_mock_target_identifier

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CONV_ID = str(uuid.uuid4())


def _make_text_message(text: str = "hello", role: str = "user") -> Message:
    return MessagePiece(
        role=role,  # type: ignore[arg-type]
        original_value=text,
        converted_value=text,
        conversation_id=_CONV_ID,
    ).to_message()


def _make_function_call_message(name: str, call_id: str = "call_1", args: dict[str, Any] | None = None) -> Message:
    value = json.dumps(
        {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(args or {}),
        }
    )
    return MessagePiece(
        role="assistant",
        original_value=value,
        original_value_data_type="function_call",  # type: ignore[arg-type]
        conversation_id=_CONV_ID,
    ).to_message()


async def _greet_fn(args: dict[str, Any]) -> dict[str, Any]:
    return {"greeting": f"Hello, {args.get('name', 'World')}!"}


_GREET_TOOL = Tool(
    name="greet",
    description="Greets someone by name",
    json_schema={},
    callable=_greet_fn,
)


def _make_tool_using_inner_mock() -> MagicMock:
    target = MagicMock(spec=PromptTarget)
    target.get_identifier.return_value = get_mock_target_identifier("InnerToolTarget")
    target.capabilities = TargetCapabilities(
        supports_tool_usage=True,
        tool_usage_schema=ToolUsageSchema(),
        supports_multi_turn=True,
    )
    return target


# ---------------------------------------------------------------------------
# S3.5: Agent IS-A PromptTarget
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_central_database")
def test_agent_is_instance_of_prompt_target() -> None:
    """Agent satisfies isinstance(agent, PromptTarget)."""
    inner = MockPromptTarget()
    agent = Agent(target=inner, toolset={_GREET_TOOL})
    assert isinstance(agent, PromptTarget)


@pytest.mark.usefixtures("patch_central_database")
def test_agent_has_send_prompt_async() -> None:
    """Agent exposes the public send_prompt_async interface."""
    inner = MockPromptTarget()
    agent = Agent(target=inner, toolset=set())
    assert callable(getattr(agent, "send_prompt_async", None))


@pytest.mark.usefixtures("patch_central_database")
def test_agent_accepted_by_prompt_sending_attack() -> None:
    """PromptSendingAttack accepts an Agent as objective_target without error."""
    inner = MockPromptTarget()
    agent = Agent(target=inner, toolset={_GREET_TOOL})
    # Should construct without raising
    attack = PromptSendingAttack(objective_target=agent)
    assert attack.get_objective_target() is agent


@pytest.mark.usefixtures("patch_central_database")
async def test_prompt_sending_attack_end_to_end_with_agent() -> None:
    """PromptSendingAttack._perform_async works end-to-end with an Agent as target."""

    inner_mock = _make_tool_using_inner_mock()
    runtime = InProcessRuntime(tools=[_GREET_TOOL])
    agent = Agent(target=inner_mock, toolset={_GREET_TOOL}, dispatcher=runtime)

    final_msg = _make_text_message("Hello, Alice!", role="assistant")

    context = SingleTurnAttackContext(
        params=AttackParameters(objective="Say hello to Alice"),
        conversation_id=str(uuid.uuid4()),
    )

    attack = PromptSendingAttack(objective_target=agent)

    # Patch the normalizer so we don't need real DB/network
    with patch.object(
        attack._prompt_normalizer,
        "send_prompt_async",
        new_callable=AsyncMock,
        return_value=final_msg,
    ):
        result = await attack._perform_async(context=context)

    # Attack should complete and return an AttackResult
    assert result is not None


@pytest.mark.usefixtures("patch_central_database")
def test_agent_get_identifier_returns_component_identifier() -> None:
    """Agent.get_identifier returns a ComponentIdentifier with Agent as class."""
    from pyrit.models import ComponentIdentifier

    inner = MockPromptTarget()
    agent = Agent(target=inner, toolset={_GREET_TOOL})
    ident = agent.get_identifier()
    assert isinstance(ident, ComponentIdentifier)
    assert ident.class_name == "Agent"


@pytest.mark.usefixtures("patch_central_database")
def test_agent_capabilities_returns_target_capabilities() -> None:
    """Agent.capabilities returns a TargetCapabilities instance."""
    inner = MockPromptTarget()
    agent = Agent(target=inner, toolset=set())
    caps = agent.capabilities
    assert isinstance(caps, TargetCapabilities)
