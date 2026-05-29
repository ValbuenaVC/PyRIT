# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from typing import Any

from httpx import HTTPStatusError

from pyrit.common import default_values, net_utility
from pyrit.common.deprecation import print_deprecation_message
from pyrit.exceptions import (
    EmptyResponseException,
    RateLimitException,
    handle_bad_request_exception,
    pyrit_target_retry,
)
from pyrit.identifiers import ComponentIdentifier
from pyrit.message_normalizer import ChatMessageNormalizer, MessageListNormalizer
from pyrit.models import (
    Message,
    MessagePiece,
    construct_response_from_request,
)
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.prompt_target.common.target_capabilities import (
    CapabilityHandlingPolicy,
    CapabilityName,
    TargetCapabilities,
    UnsupportedCapabilityBehavior,
)
from pyrit.prompt_target.common.target_configuration import TargetConfiguration
from pyrit.prompt_target.common.utils import limit_requests_per_minute, validate_temperature, validate_top_p
from pyrit.tools import ToolBackend, ToolCallParser

logger = logging.getLogger(__name__)


class AzureMLChatTarget(PromptTarget):
    """
    A prompt target for Azure Machine Learning chat endpoints.

    This class works with most chat completion Instruct models deployed on Azure AI Machine Learning
    Studio endpoints (including but not limited to: mistralai-Mixtral-8x7B-Instruct-v01,
    mistralai-Mistral-7B-Instruct-v01, Phi-3.5-MoE-instruct, Phi-3-mini-4k-instruct,
    Llama-3.2-3B-Instruct, and Meta-Llama-3.1-8B-Instruct).

    Please create or adjust environment variables (endpoint and key) as needed for the model you are using.
    """

    endpoint_uri_environment_variable: str = "AZURE_ML_MANAGED_ENDPOINT"
    api_key_environment_variable: str = "AZURE_ML_KEY"

    _DEFAULT_CONFIGURATION: TargetConfiguration = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_message_pieces=True,
            supports_editable_history=True,
            supports_multi_turn=True,
            supports_system_prompt=True,
        )
    )

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        model_name: str = "",
        message_normalizer: MessageListNormalizer[Any] | None = None,
        max_new_tokens: int = 400,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        max_requests_per_minute: int | None = None,
        custom_configuration: TargetConfiguration | None = None,
        tool_parser: ToolCallParser | None = None,
        tool_backend: ToolBackend | None = None,
        **param_kwargs: Any,
    ) -> None:
        """
        Initialize an instance of the AzureMLChatTarget class.

        Args:
            endpoint (str | None): The endpoint URL for the deployed Azure ML model.
                Defaults to the value of the AZURE_ML_MANAGED_ENDPOINT environment variable.
            api_key (str | None): The API key for accessing the Azure ML endpoint.
                Defaults to the value of the `AZURE_ML_KEY` environment variable.
            model_name (str): The name of the model being used (e.g., "Llama-3.2-3B-Instruct").
                Used for identification purposes. Defaults to empty string.
            message_normalizer (MessageListNormalizer[Any] | None): **Deprecated.** Use
                ``custom_configuration`` with ``CapabilityHandlingPolicy`` instead. Previously used for
                models that do not allow system prompts.
                Will be removed in 0.15.0.
            max_new_tokens (int): The maximum number of tokens to generate in the response.
                Defaults to 400.
            temperature (float): The temperature for generating diverse responses. 1.0 is most random,
                0.0 is least random. Defaults to 1.0.
            top_p (float): The top-p value for generating diverse responses. It represents
                the cumulative probability of the top tokens to keep. Defaults to 1.0.
            repetition_penalty (float): The repetition penalty for generating diverse responses.
                1.0 means no penalty with a greater value (up to 2.0) meaning more penalty for repeating tokens.
                Defaults to 1.2.
            max_requests_per_minute (int | None): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
            custom_configuration (TargetConfiguration | None): Override the default configuration for this target
                instance. Useful for targets whose capabilities depend on deployment configuration.
            tool_parser (ToolCallParser | None): When supplied, the target opts into PyRIT's
                ``@tool_loop`` and uses this parser to extract pending tool calls from the
                response. Supplying a parser also enables the ``supports_tool_use`` capability
                on the default configuration so callers don't have to construct a custom
                configuration just to enable the loop. The parser's expectations about the
                deployment's response shape MUST line up with the contract documented in
                ``doc/code/targets/`` for tool-capable Azure ML deployments.
            tool_backend (ToolBackend | None): Convenience kwarg that wires a tool backend
                onto ``custom_configuration.tool_backend``. Equivalent to constructing a
                ``TargetConfiguration`` with the backend assigned. When ``custom_configuration``
                already specifies a backend, the kwarg is rejected.
            **param_kwargs: Additional parameters to pass to the model for generating responses. Example
                parameters can be found here: https://huggingface.co/docs/api-inference/tasks/text-generation.
                Note that the link above may not be comprehensive, and specific acceptable parameters may be
                model-dependent. If a model does not accept a certain parameter that is passed in, it will be skipped
                without throwing an error.

        Raises:
            ValueError: If both `message_normalizer` and `custom_configuration` are provided,
                since `message_normalizer` is deprecated and the two configurations may conflict.
        """
        endpoint_value = default_values.get_required_value(
            env_var_name=self.endpoint_uri_environment_variable, passed_value=endpoint
        )

        # Translate legacy message_normalizer into TargetConfiguration
        if message_normalizer is not None:
            if custom_configuration is not None:
                raise ValueError(
                    "Cannot specify both 'message_normalizer' and 'custom_configuration'. "
                    "Use 'custom_configuration' only; 'message_normalizer' is deprecated and "
                    "will be removed in 0.15.0."
                )
            print_deprecation_message(
                old_item="AzureMLChatTarget(message_normalizer=...)",
                new_item="AzureMLChatTarget(custom_configuration=...)",
                removed_in="0.15.0",
            )
            # The legacy message_normalizer was primarily used to handle system prompts
            # for models that don't support them (e.g. GenericSystemSquashNormalizer).
            # We translate it into a TargetConfiguration that marks system_prompt as
            # unsupported + ADAPT so the pipeline invokes the user's normalizer.
            default_caps = self._DEFAULT_CONFIGURATION.capabilities
            default_behaviors = dict(self._DEFAULT_CONFIGURATION.policy.behaviors)
            default_behaviors[CapabilityName.SYSTEM_PROMPT] = UnsupportedCapabilityBehavior.ADAPT
            custom_configuration = TargetConfiguration(
                capabilities=TargetCapabilities(
                    supports_multi_message_pieces=default_caps.supports_multi_message_pieces,
                    supports_editable_history=default_caps.supports_editable_history,
                    supports_multi_turn=default_caps.supports_multi_turn,
                    supports_system_prompt=False,
                ),
                policy=CapabilityHandlingPolicy(behaviors=default_behaviors),
                normalizer_overrides={CapabilityName.SYSTEM_PROMPT: message_normalizer},
            )

        # Enable tool-use capability when a parser is supplied so callers
        # don't need to construct a custom_configuration just to opt in.
        if tool_parser is not None:
            custom_configuration = self._enable_tool_use(configuration=custom_configuration)

        # tool_backend is a convenience kwarg; install it into the configuration.
        if tool_backend is not None:
            custom_configuration = self._install_tool_backend(
                configuration=custom_configuration,
                tool_backend=tool_backend,
            )

        PromptTarget.__init__(
            self,
            max_requests_per_minute=max_requests_per_minute,
            endpoint=endpoint_value,
            model_name=model_name,
            custom_configuration=custom_configuration,
        )

        self._initialize_vars(endpoint=endpoint, api_key=api_key)

        validate_temperature(temperature)
        validate_top_p(top_p)

        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty
        self._extra_parameters = param_kwargs
        self._tool_parser_instance = tool_parser

    def _enable_tool_use(self, *, configuration: TargetConfiguration | None) -> TargetConfiguration:
        """
        Return a configuration whose capabilities include ``supports_tool_use=True``.

        When ``configuration`` already has the capability set, returns it as-is.
        Otherwise rebuilds the capabilities with ``supports_tool_use=True`` flipped
        on and preserves every other field.

        Args:
            configuration (TargetConfiguration | None): The user-supplied configuration,
                or ``None`` to start from the class default.

        Returns:
            TargetConfiguration: A configuration whose capabilities include
                ``supports_tool_use=True``.
        """
        source = configuration if configuration is not None else self._DEFAULT_CONFIGURATION
        caps = source.capabilities
        if caps.includes(capability=CapabilityName.TOOL_USE):
            return source
        updated_caps = TargetCapabilities(
            supports_multi_message_pieces=caps.supports_multi_message_pieces,
            supports_editable_history=caps.supports_editable_history,
            supports_multi_turn=caps.supports_multi_turn,
            supports_system_prompt=caps.supports_system_prompt,
            supports_tool_use=True,
            input_modalities=caps.input_modalities,
            output_modalities=caps.output_modalities,
        )
        return TargetConfiguration(
            capabilities=updated_caps,
            policy=source.policy,
            tool_event_policy=source.tool_event_policy,
            tool_backend=source.tool_backend,
        )

    @staticmethod
    def _install_tool_backend(
        *,
        configuration: TargetConfiguration | None,
        tool_backend: ToolBackend,
    ) -> TargetConfiguration:
        """
        Install ``tool_backend`` onto ``configuration``. Rejects double-supply.

        Args:
            configuration (TargetConfiguration | None): The user-supplied configuration.
            tool_backend (ToolBackend): The backend to install.

        Returns:
            TargetConfiguration: The same ``configuration`` instance with the
                backend installed.

        Raises:
            ValueError: When ``configuration`` is ``None`` (no capability to attach
                to), or when ``configuration.tool_backend`` is already set to a
                different backend.
        """
        if configuration is None:
            raise ValueError(
                "tool_backend kwarg requires capabilities.supports_tool_use=True; "
                "supply tool_parser= so the default capabilities flip TOOL_USE on, "
                "or build a custom_configuration explicitly."
            )
        if configuration.tool_backend is not None and configuration.tool_backend is not tool_backend:
            raise ValueError("tool_backend kwarg conflicts with custom_configuration.tool_backend; supply only one.")
        configuration.tool_backend = tool_backend
        return configuration

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build the identifier with Azure ML-specific parameters.

        Returns:
            ComponentIdentifier: The identifier for this target instance.
        """
        return self._create_identifier(
            params={
                "temperature": self._temperature,
                "top_p": self._top_p,
                "max_new_tokens": self._max_new_tokens,
                "repetition_penalty": self._repetition_penalty,
            },
        )

    def _initialize_vars(self, endpoint: str | None = None, api_key: str | None = None) -> None:
        """
        Set the endpoint and key for accessing the Azure ML model. Use this function to manually
        pass in your own endpoint uri and api key. Defaults to the values in the .env file for the variables
        stored in self.endpoint_uri_environment_variable and self.api_key_environment_variable (which default to
        "AZURE_ML_MANAGED_ENDPOINT" and "AZURE_ML_KEY" respectively). It is recommended to set these variables
        in the .env file and call _set_env_configuration_vars rather than passing the uri and key directly to
        this function or the target constructor.

        Args:
            endpoint (str, optional): The endpoint uri for the deployed Azure ML model.
            api_key (str, optional): The API key for accessing the Azure ML endpoint.
        """
        self._endpoint = default_values.get_required_value(
            env_var_name=self.endpoint_uri_environment_variable, passed_value=endpoint
        )
        self._api_key = default_values.get_required_value(
            env_var_name=self.api_key_environment_variable, passed_value=api_key
        )

    @limit_requests_per_minute
    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        """
        Asynchronously send a message to the Azure ML chat target.

        Args:
            normalized_conversation (list[Message]): The full conversation
                (history + current message) after running the normalization
                pipeline. The current message is the last element.

        Returns:
            list[Message]: A list containing the response from the prompt target.

        Raises:
            EmptyResponseException: If the response from the chat is empty.
            RateLimitException: If the target rate limit is exceeded.
            HTTPStatusError: For any other HTTP errors during the process.
        """
        message = normalized_conversation[-1]
        request = message.message_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        try:
            response_body = await self._complete_chat_async(messages=normalized_conversation)
            response_entry = self._materialize_response(response=response_body, request=request)
        except HTTPStatusError as hse:
            if hse.response.status_code == 400:
                response_entry = handle_bad_request_exception(response_text=hse.response.text, request=request)
            elif hse.response.status_code == 429:
                raise RateLimitException from hse
            else:
                raise hse

        logger.info("Received the following response from the prompt target" + f"{response_entry.get_value()}")
        return [response_entry]

    @pyrit_target_retry
    async def _complete_chat_async(
        self,
        messages: list[Message],
    ) -> dict[str, Any]:
        """
        Issue a single chat request and return the parsed JSON response body.

        Args:
            messages (list[Message]): The message objects containing the role and content.

        Returns:
            dict[str, Any]: The deserialized response body. Always includes an
                ``output`` field (per the AML scoring-script contract). Tool-capable
                deployments may additionally include a ``tool_calls`` field carrying
                canonical envelopes.

        Raises:
            EmptyResponseException: If the response from the chat is empty.
            ValueError: If the parsed response body is missing the ``output`` field.
            Exception: For any other errors during the process.
        """
        headers = self._get_headers()
        payload = await self._construct_http_body_async(messages)

        response = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=self._endpoint, method="POST", request_body=payload, headers=headers
        )

        body = response.json()
        if not isinstance(body, dict) or body == {}:
            raise EmptyResponseException(message="The chat returned an empty response.")
        if "output" not in body:
            raise ValueError(f"Response from the target did not include 'output'. Returned response: {body}.")
        return body

    def _materialize_response(self, *, response: dict[str, Any], request: MessagePiece) -> Message:
        """
        Build a ``Message`` from the parsed response body, handling tool calls.

        The deployment may include a ``tool_calls`` list when the model emits
        canonical envelopes. Each envelope becomes its own ``function_call``
        MessagePiece so the ``CanonicalEnvelopeParser`` shipped with PyRIT can
        recognize it without further translation.

        Args:
            response (dict[str, Any]): The parsed response body returned from the endpoint.
            request (MessagePiece): The request piece used to stamp identity onto each
                response piece.

        Returns:
            Message: The materialized response message. Has at least one piece;
                when both ``output`` and ``tool_calls`` are present, the text piece
                comes first followed by one function_call piece per envelope.

        Raises:
            EmptyResponseException: If the response has neither output text nor tool calls.
        """
        text = str(response.get("output") or "")
        tool_envelopes = response.get("tool_calls") or []
        if not text and not tool_envelopes:
            raise EmptyResponseException(message="The chat returned an empty response.")

        pieces: list[MessagePiece] = []
        if text:
            text_piece = construct_response_from_request(request=request, response_text_pieces=[text]).message_pieces[0]
            pieces.append(text_piece)
        for envelope in tool_envelopes:
            fc_piece = construct_response_from_request(
                request=request,
                response_text_pieces=[json.dumps(envelope, separators=(",", ":"))],
                response_type="function_call",
            ).message_pieces[0]
            pieces.append(fc_piece)
        return Message(message_pieces=pieces, skip_validation=True)

    async def _construct_http_body_async(
        self,
        messages: list[Message],
    ) -> dict[str, Any]:
        """
        Construct the HTTP request body for the AML online endpoint.

        Args:
            messages: List of chat messages to include in the request body.

        Returns:
            dict: The constructed HTTP request body.
        """
        wire_format = ChatMessageNormalizer()
        messages_dict = await wire_format.normalize_to_dicts_async(messages)

        body: dict[str, Any] = {
            "input_data": {
                "input_string": messages_dict,
                "parameters": {
                    "max_new_tokens": self._max_new_tokens,
                    "temperature": self._temperature,
                    "top_p": self._top_p,
                    "repetition_penalty": self._repetition_penalty,
                }
                | self._extra_parameters,
            }
        }
        schemas = self._tool_schemas()
        if schemas:
            body["tools"] = schemas
        return body

    @property
    def _tool_parser(self) -> ToolCallParser | None:
        """Return the parser supplied at construction, if any."""
        return self._tool_parser_instance

    def _tool_schemas(self) -> list[dict[str, Any]]:
        """
        Wrap the backend's schemas in the OpenAI Chat Completions ``tools`` shape.

        Tool-capable deployments are expected to forward ``tools`` into
        ``tokenizer.apply_chat_template`` after unwrapping the ``{"type":
        "function", "function": {...}}`` envelope.

        Returns:
            list[dict[str, Any]]: One descriptor per advertised tool, or an
                empty list when no backend is configured.
        """
        return [{"type": "function", "function": schema} for schema in super()._tool_schemas()]

    def _get_headers(self) -> dict[str, str]:
        """
        Headers for accessing inference endpoint deployed in AML.

        Returns:
            headers(dict): contains bearer token as AML key and content-type: JSON
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self._api_key),
        }

        return headers

    def _validate_request(self, *, normalized_conversation: list[Message]) -> None:
        pass
