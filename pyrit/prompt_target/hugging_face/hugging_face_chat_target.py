# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, cast

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PretrainedConfig,
)

from pyrit.common import default_values
from pyrit.common.download_hf_model import download_specific_files_async
from pyrit.exceptions import EmptyResponseException, pyrit_target_retry
from pyrit.identifiers import ComponentIdentifier
from pyrit.models import Message, construct_response_from_request
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.prompt_target.common.target_capabilities import (
    CapabilityName,
    TargetCapabilities,
)
from pyrit.prompt_target.common.target_configuration import TargetConfiguration
from pyrit.prompt_target.common.utils import limit_requests_per_minute
from pyrit.tools import ToolBackend, ToolCallParser

logger = logging.getLogger(__name__)


class HuggingFaceChatTarget(PromptTarget):
    """
    The HuggingFaceChatTarget interacts with HuggingFace models, specifically for conducting red teaming activities.
    Inherits from PromptTarget to comply with the current design standards.
    """

    _DEFAULT_CONFIGURATION: TargetConfiguration = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_editable_history=True,
            supports_system_prompt=True,
        )
    )

    # Class-level cache for model and tokenizer
    _cached_model: Any = None
    _cached_tokenizer: Any = None
    _cached_model_id: str | None = None

    # Class-level flag to enable or disable cache
    _cache_enabled = True

    # Define the environment variable name for the Hugging Face token
    HUGGINGFACE_TOKEN_ENVIRONMENT_VARIABLE = "HUGGINGFACE_TOKEN"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        model_path: str | None = None,
        hf_access_token: str | None = None,
        use_cuda: bool = False,
        tensor_format: str = "pt",
        necessary_files: list[str] | None = None,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        do_sample: bool | None = None,
        repetition_penalty: float | None = None,
        random_seed: int | None = None,
        skip_special_tokens: bool = True,
        trust_remote_code: bool = False,
        device_map: str | None = None,
        torch_dtype: Any | None = None,
        attn_implementation: str | None = None,
        max_requests_per_minute: int | None = None,
        custom_configuration: TargetConfiguration | None = None,
        tool_parser: ToolCallParser | None = None,
        tool_backend: ToolBackend | None = None,
    ) -> None:
        """
        Initialize the HuggingFaceChatTarget.

        Args:
            model_id (str | None): The Hugging Face model ID. Either model_id or model_path must be provided.
            model_path (str | None): Path to a local model. Either model_id or model_path must be provided.
            hf_access_token (str | None): Hugging Face access token for authentication.
            use_cuda (bool): Whether to use CUDA for GPU acceleration. Defaults to False.
            tensor_format (str): The tensor format. Defaults to "pt".
            necessary_files (list[str] | None): List of necessary model files to download.
            max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 20.
            temperature (float): Sampling temperature. Defaults to 1.0.
            top_p (float): Nucleus sampling probability. Defaults to 1.0.
            top_k (int | None): Top-K sampling parameter. Only used when do_sample is True.
                Defaults to None (uses model default, typically 50).
            do_sample (bool | None): Whether to use sampling instead of greedy decoding. When None,
                sampling is automatically enabled if temperature, top_p, or top_k suggest
                non-greedy decoding. Defaults to None.
            repetition_penalty (float | None): Penalty for repeating tokens. Values > 1.0 discourage
                repetition. Defaults to None (uses model default, typically 1.0).
            random_seed (int | None): Random seed for deterministic generation. When set, calls
                torch.manual_seed() at construction time. Defaults to None.
            skip_special_tokens (bool): Whether to skip special tokens. Defaults to True.
            trust_remote_code (bool): Whether to trust remote code execution. Defaults to False.
            device_map (str | None): Device mapping strategy.
            torch_dtype (Any | None): Torch data type for model weights.
            attn_implementation (str | None): Attention implementation type.
            max_requests_per_minute (int | None): The maximum number of requests per minute. Defaults to None.
            custom_configuration (TargetConfiguration | None): Override the default configuration for this target
                instance. Defaults to None.
            tool_parser (ToolCallParser | None): When supplied, the target opts into PyRIT's
                ``@tool_loop`` and uses this parser to extract pending tool calls from each
                generated response. Supplying a parser also enables ``supports_tool_use=True``
                on the default capabilities so callers don't need a custom_configuration just
                to opt in. ``InlineToolCallParser`` is the typical choice because the local
                tokenizer emits tool calls as inline marker-delimited JSON; supply a different
                parser when targeting a chat template with a different marker syntax.
            tool_backend (ToolBackend | None): Convenience kwarg that installs the backend
                onto ``custom_configuration.tool_backend``.

        Raises:
            ValueError: If neither or both of `model_id` and `model_path` are provided.
            RuntimeError: If torch cannot be imported or if CUDA is requested but not available.
        """
        model_name = model_id if model_id else model_path if model_path else ""

        # Enable tool-use capability when a parser is supplied, BEFORE super().__init__
        # so the configuration is correct by the time the base class records it.
        if tool_parser is not None:
            custom_configuration = self._enable_tool_use(configuration=custom_configuration)
        if tool_backend is not None:
            custom_configuration = self._install_tool_backend(
                configuration=custom_configuration,
                tool_backend=tool_backend,
            )

        super().__init__(
            max_requests_per_minute=max_requests_per_minute,
            model_name=model_name,
            custom_configuration=custom_configuration,
        )

        if not model_id and not model_path:
            raise ValueError("Either `model_id` or `model_path` must be provided.")
        if model_id and model_path:
            raise ValueError("Provide only one of `model_id` or `model_path`, not both.")

        self.model_id = model_id
        self.model_path = model_path
        self.use_cuda = use_cuda
        self.tensor_format = tensor_format
        self.trust_remote_code = trust_remote_code
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation

        # Only get the Hugging Face token if a model ID is provided
        if model_id:
            self.huggingface_token = default_values.get_required_value(
                env_var_name=self.HUGGINGFACE_TOKEN_ENVIRONMENT_VARIABLE, passed_value=hf_access_token
            )
        else:
            self.huggingface_token = None

        try:
            import torch
        except ModuleNotFoundError as e:
            raise RuntimeError("Could not import torch. You may need to install it via 'pip install pyrit[all]'") from e

        # Determine the device
        self.device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Set necessary files if provided, otherwise set to None to trigger general download
        self.necessary_files = necessary_files

        # Set the default parameters for the model generation
        self.max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._do_sample = do_sample
        self._repetition_penalty = repetition_penalty
        self._random_seed = random_seed
        self.skip_special_tokens = skip_special_tokens

        self._warn_if_sampling_params_without_do_sample()

        self._generation_params = self._build_generation_params()
        self._seed_rng()

        if self.use_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

        self.load_model_and_tokenizer_task = asyncio.create_task(self.load_model_and_tokenizer())
        self._tool_parser_instance = tool_parser

    @classmethod
    def _enable_tool_use(cls, *, configuration: TargetConfiguration | None) -> TargetConfiguration:
        """
        Return a configuration whose capabilities include ``supports_tool_use=True``.

        When ``configuration`` already has the capability set, returns it as-is.
        Otherwise rebuilds the capabilities with ``supports_tool_use=True`` and
        preserves every other field.

        Args:
            configuration (TargetConfiguration | None): The user-supplied configuration,
                or ``None`` to start from the class default.

        Returns:
            TargetConfiguration: A configuration whose capabilities include
                ``supports_tool_use=True``.
        """
        source = configuration if configuration is not None else cls._DEFAULT_CONFIGURATION
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
            ValueError: When ``configuration`` is ``None``, or when
                ``configuration.tool_backend`` is already set to a different backend.
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

    @property
    def _tool_parser(self) -> ToolCallParser | None:
        """Return the parser supplied at construction, if any."""
        return self._tool_parser_instance

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build the identifier with HuggingFace chat-specific parameters.

        Returns:
            ComponentIdentifier: The identifier for this target instance.
        """
        return self._create_identifier(
            params={
                "temperature": self._temperature,
                "top_p": self._top_p,
                "top_k": self._top_k,
                "do_sample": self._do_sample,
                "repetition_penalty": self._repetition_penalty,
                "random_seed": self._random_seed,
                "max_new_tokens": self.max_new_tokens,
                "skip_special_tokens": self.skip_special_tokens,
                "use_cuda": self.use_cuda,
                "tensor_format": self.tensor_format,
                "trust_remote_code": self.trust_remote_code,
                "device_map": self.device_map,
                "torch_dtype": str(self.torch_dtype) if self.torch_dtype else None,
                "attn_implementation": self.attn_implementation,
            },
        )

    def _load_from_path(self, path: str, **kwargs: Any) -> None:
        """
        Load the model and tokenizer from a given path.

        Args:
            path: The path to load the model and tokenizer from.
            **kwargs: Additional keyword arguments to pass to the model loader.
        """
        logger.info(f"Loading model and tokenizer from path: {path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=self.trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=self.trust_remote_code, **kwargs)

    def is_model_id_valid(self) -> bool:
        """
        Check if the HuggingFace model ID is valid.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            # Attempt to load the configuration of the model
            PretrainedConfig.from_pretrained(self.model_id or "")
            return True
        except Exception as e:
            logger.error(f"Invalid HuggingFace model ID {self.model_id}: {e}")
            return False

    async def load_model_and_tokenizer(self) -> None:
        """
        Load the model and tokenizer, download if necessary.

        Downloads the model to the HF_MODELS_DIR folder if it does not exist,
        then loads it from there.

        Raises:
            Exception: If the model loading fails.
        """
        try:
            # Determine the identifier for caching purposes
            model_identifier = self.model_path or self.model_id

            optional_model_kwargs = {
                key: value
                for key, value in {
                    "device_map": self.device_map,
                    "torch_dtype": self.torch_dtype,
                    "attn_implementation": self.attn_implementation,
                }.items()
                if value is not None
            }

            # Check if the model is already cached
            if HuggingFaceChatTarget._cache_enabled and HuggingFaceChatTarget._cached_model_id == model_identifier:
                logger.info(f"Using cached model and tokenizer for {model_identifier}.")
                self.model = HuggingFaceChatTarget._cached_model
                self.tokenizer = HuggingFaceChatTarget._cached_tokenizer
                return

            if self.model_path:
                # Load the tokenizer and model from the local directory
                logger.info(f"Loading model from local path: {self.model_path}...")
                self._load_from_path(self.model_path, **optional_model_kwargs)
            else:
                # Define the default Hugging Face cache directory
                cache_dir = os.path.join(
                    os.path.expanduser("~"),
                    ".cache",
                    "huggingface",
                    "hub",
                    f"models--{(self.model_id or '').replace('/', '--')}",
                )

                if self.necessary_files is None:
                    # Download all files if no specific files are provided
                    logger.info(f"Downloading all files for {self.model_id}...")
                    await download_specific_files_async(
                        self.model_id or "",
                        None,
                        self.huggingface_token,  # type: ignore[ty:invalid-argument-type]
                        Path(cache_dir),
                    )
                else:
                    # Download only the necessary files
                    logger.info(f"Downloading specific files for {self.model_id}...")
                    await download_specific_files_async(
                        self.model_id or "",
                        self.necessary_files,
                        self.huggingface_token,  # type: ignore[ty:invalid-argument-type]
                        Path(cache_dir),
                    )

                # Load the tokenizer and model from the specified directory
                logger.info(f"Loading model {self.model_id} from cache path: {cache_dir}...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id or "", cache_dir=cache_dir, trust_remote_code=self.trust_remote_code
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id or "",
                    cache_dir=cache_dir,
                    trust_remote_code=self.trust_remote_code,
                    **optional_model_kwargs,
                )

            # Move the model to the correct device
            self.model = self.model.to(self.device)

            # Debug prints to check types
            logger.info(f"Model loaded: {type(self.model)}")
            logger.info(f"Tokenizer loaded: {type(self.tokenizer)}")

            # Cache the loaded model and tokenizer if caching is enabled
            if HuggingFaceChatTarget._cache_enabled:
                HuggingFaceChatTarget._cached_model = self.model
                HuggingFaceChatTarget._cached_tokenizer = self.tokenizer
                HuggingFaceChatTarget._cached_model_id = model_identifier

            logger.info(f"Model {model_identifier} loaded successfully.")

        except Exception as e:
            logger.error(f"Error loading model {self.model_id}: {e}")
            raise

    @limit_requests_per_minute
    @pyrit_target_retry
    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        """
        Send a normalized prompt asynchronously to the HuggingFace model.

        Builds the full chat history (system, user, assistant turns) from the normalized
        conversation and passes it through the model's chat template.

        Args:
            normalized_conversation (list[Message]): The full conversation
                (history + current message) after running the normalization
                pipeline. The current message is the last element.

        Returns:
            list[Message]: A list containing the response object with generated text pieces.

        Raises:
            EmptyResponseException: If the model generates an empty response.
        """
        await self.load_model_and_tokenizer_task

        request = normalized_conversation[-1].message_pieces[0]

        messages = self._build_chat_messages(normalized_conversation=normalized_conversation)

        logger.info(f"Sending the following messages to the HuggingFace model: {messages}")

        tokenized_chat = self._apply_chat_template(messages)
        input_ids = tokenized_chat["input_ids"]
        attention_mask = tokenized_chat["attention_mask"]

        logger.info(f"Tokenized chat: {input_ids}")

        try:
            # Ensure model is on the correct device (should already be, but safeguard for device changes)
            self.model.to(self.device)

            # Record input length to extract only newly generated tokens
            input_length = input_ids.shape[-1]

            generate_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, **self._generation_params}

            logger.info("Generating response from model...")
            generated_ids = self.model.generate(**generate_kwargs)

            logger.info(f"Generated IDs: {generated_ids}")

            generated_tokens = generated_ids[0][input_length:]

            assistant_response = cast(
                "str",
                self.tokenizer.decode(generated_tokens, skip_special_tokens=self.skip_special_tokens),
            ).strip()

            if not assistant_response:
                raise EmptyResponseException

            logger.info(f"Assistant's response: {assistant_response}")

            model_identifier = self.model_id or self.model_path

            effective_config = self._get_effective_generation_config()

            response = construct_response_from_request(
                request=request,
                response_text_pieces=[assistant_response],
                prompt_metadata={
                    "model_id": model_identifier or "",
                    "effective_generation_config": json.dumps(effective_config, default=str),
                },
            )
            return [response]

        except Exception as e:
            logger.error(f"Error occurred during inference: {e}")
            raise

    def _build_chat_messages(self, *, normalized_conversation: list[Message]) -> list[dict[str, Any]]:
        """
        Build a list of chat message dicts from the full normalized conversation.

        Includes system, user, and assistant messages from the conversation history
        so that the model's chat template receives the complete context. When the
        conversation contains tool-call envelopes (produced by ``@tool_loop``), they
        are converted into the chat-template's tool message shape:

        * ``function_call`` pieces become an ``assistant`` message with a
          ``tool_calls`` list (matching the HuggingFace ``apply_chat_template``
          convention; templates that don't recognize ``tool_calls`` fall back to
          rendering the embedded JSON as content).
        * ``function_call_output`` pieces become a ``role=tool`` message with the
          tool result as content and ``tool_call_id`` carried for templates that
          need it.

        Args:
            normalized_conversation (list[Message]): The full normalized conversation.

        Returns:
            list[dict[str, Any]]: Messages formatted for the chat template.
        """
        messages: list[dict[str, Any]] = []
        for msg in normalized_conversation:
            for piece in msg.message_pieces:
                tool_dict = self._maybe_tool_chat_message(piece=piece)
                if tool_dict is not None:
                    messages.append(tool_dict)
                    continue
                role = piece.api_role
                content = piece.converted_value or ""
                messages.append({"role": role, "content": content})
        return messages

    @staticmethod
    def _maybe_tool_chat_message(*, piece: Any) -> dict[str, Any] | None:
        """
        Convert a ``function_call`` or ``function_call_output`` piece to a chat-template message.

        Args:
            piece (Any): The MessagePiece to inspect.

        Returns:
            dict[str, Any] | None: A chat-template message dict (``assistant`` with
                ``tool_calls``, or ``role=tool`` with ``tool_call_id``) when the
                piece carries a tool envelope, otherwise ``None``.
        """
        data_type = piece.converted_value_data_type or piece.original_value_data_type
        if data_type not in ("function_call", "function_call_output"):
            return None
        envelope = json.loads(piece.converted_value)
        if data_type == "function_call":
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": envelope.get("call_id", ""),
                        "type": "function",
                        "function": {
                            "name": envelope.get("name", ""),
                            "arguments": envelope.get("arguments", "{}"),
                        },
                    }
                ],
            }
        return {
            "role": "tool",
            "content": str(envelope.get("output", "")),
            "tool_call_id": envelope.get("call_id", ""),
        }

    def set_random_seed(self, random_seed: int) -> None:
        """
        Set a new random seed and immediately re-seed the RNG.

        Allows re-seeding between conversations or experiments for controlled
        reproducibility. The initial seed (if any) is applied once at construction
        time; call this method to change it later.

        Args:
            random_seed (int): The random seed value.
        """
        self._random_seed = random_seed
        self._seed_rng()

    def _build_generation_params(self) -> dict[str, Any]:
        """
        Build the static generation parameters dict.

        Computed once at init. Only includes optional parameters when they
        are explicitly set (not None), allowing the model's own
        generation_config defaults to apply otherwise.

        Returns:
            dict[str, Any]: Static keyword arguments for model.generate().
        """
        params: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
        }
        if self._top_k is not None:
            params["top_k"] = self._top_k
        if self._do_sample is not None:
            params["do_sample"] = self._do_sample
        if self._repetition_penalty is not None:
            params["repetition_penalty"] = self._repetition_penalty
        return params

    def _seed_rng(self) -> None:
        """
        Seed the random number generators for deterministic generation.

        When ``self._random_seed`` is set, seeds both CPU and CUDA RNGs before each
        ``model.generate()`` call. This enables reproducible results when all other
        parameters are held constant.

        Note:
            This sets global torch RNG state. Concurrent generation calls on
            the same process may interfere with determinism.
        """
        if self._random_seed is not None:
            import torch

            torch.manual_seed(self._random_seed)
            if self.use_cuda:
                torch.cuda.manual_seed_all(self._random_seed)

    def _get_effective_generation_config(self) -> dict[str, Any]:
        """
        Return the effective generation parameters that were used for the last call.

        Combines the model's own generation_config with the explicit overrides from
        this target instance, so that the stored metadata reflects what actually ran.

        Returns:
            dict[str, Any]: Merged generation configuration.
        """
        effective: dict[str, Any] = {}
        if hasattr(self.model, "generation_config"):
            effective = self.model.generation_config.to_dict()

        effective.update(self._generation_params)
        if self._random_seed is not None:
            effective["random_seed"] = self._random_seed
        return effective

    def _warn_if_sampling_params_without_do_sample(self) -> None:
        """
        Emit a warning when sampling parameters are set but do_sample is not explicitly True.

        Sampling-specific parameters (temperature != 1.0, top_p != 1.0, top_k) are
        ignored by HuggingFace's generate() unless do_sample=True. This helps users
        avoid silent misconfiguration.
        """
        has_sampling_override = self._temperature != 1.0 or self._top_p != 1.0 or self._top_k is not None
        if has_sampling_override and self._do_sample is not True:
            warnings.warn(
                "Sampling parameters (temperature, top_p, top_k) are set but do_sample is not True. "
                "HuggingFace ignores these parameters during greedy decoding. "
                "Set do_sample=True to enable sampling.",
                UserWarning,
                stacklevel=3,
            )

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> Any:
        """
        Apply the chat template to the input messages and tokenize them.

        When ``self._tool_schemas()`` is non-empty, the schemas are forwarded
        into ``apply_chat_template`` so tool-trained chat templates can render
        the model-family-specific tools block (Qwen wraps in ``<tools>...</tools>``,
        Llama uses a system-message preamble, etc.). The model can then emit
        tool calls in its native marker syntax which the user-supplied
        ``tool_parser`` extracts.

        Args:
            messages: The input messages to apply the chat template to.

        Returns:
            dict: Tokenized inputs ready for the model.

        Raises:
            ValueError: If the tokenizer does not have a chat template.
        """
        # Check if the tokenizer has a chat template
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            logger.info("Tokenizer has a chat template. Applying it to the input messages.")

            template_kwargs: dict[str, Any] = {}
            schemas = self._tool_schemas()
            if schemas:
                template_kwargs["tools"] = schemas

            # Apply the chat template to format and tokenize the messages
            return cast(
                "BatchEncoding",
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors=self.tensor_format,
                    return_dict=True,
                    **template_kwargs,
                ),
            ).to(self.device)
        error_message = (
            "Tokenizer does not have a chat template. "
            "This model is not supported, as we only support instruct models with a chat template."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    def is_json_response_supported(self) -> bool:
        """
        Check if the target supports JSON as a response format.

        Returns:
            bool: True if JSON response is supported, False otherwise.
        """
        return self.capabilities.supports_json_output

    @classmethod
    def enable_cache(cls) -> None:
        """Enable the class-level cache."""
        cls._cache_enabled = True
        logger.info("Class-level cache enabled.")

    @classmethod
    def disable_cache(cls) -> None:
        """Disables the class-level cache and clears the cache."""
        cls._cache_enabled = False
        cls._cached_model = None
        cls._cached_tokenizer = None
        cls._cached_model_id = None
        logger.info("Class-level cache disabled and cleared.")
