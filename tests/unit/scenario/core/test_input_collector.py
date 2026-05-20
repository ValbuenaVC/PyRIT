# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Phase 8c — coverage for the ``InputCollector`` Protocol and reference impls."""

from __future__ import annotations

from typing import Any

import pytest

from pyrit.scenario.core.builder import ScenarioInputValidationError
from pyrit.scenario.core.input_collector import (
    ArtifactInputCollector,
    CliInputCollector,
    DictInputCollector,
    InputCollector,
    MaxAttemptsExceededError,
    OpaqueRoleNotElicitableError,
    collect_inputs_with_retry,
)
from pyrit.scenario.core.input_schema import RoleDescriptor, RoleTag

# --- shared fixtures -------------------------------------------------------------


def _scalar_role(
    *,
    name: str = "label",
    description: str = "desc",
    param_type: type | None = str,
    required: bool = True,
    default: Any = None,
) -> RoleDescriptor:
    return RoleDescriptor(
        name=name,
        description=description,
        tag=RoleTag.SCALAR,
        param_type=param_type,
        required=required,
        default=default,
    )


def _choice_role(*, name: str = "mode", choices: tuple[str, ...] = ("fast", "slow")) -> RoleDescriptor:
    return RoleDescriptor(
        name=name,
        description="pick one",
        tag=RoleTag.CHOICE,
        param_type=str,
        choices=choices,
    )


def _opaque_role(*, name: str = "instance") -> RoleDescriptor:
    return RoleDescriptor(
        name=name,
        description="pre-built thing",
        tag=RoleTag.OPAQUE,
    )


# --- DictInputCollector ----------------------------------------------------------


class TestDictInputCollector:
    def test_returns_value_when_present(self):
        collector = DictInputCollector({"label": "harm"})
        assert collector.collect(role=_scalar_role()) == "harm"

    def test_missing_required_raises_with_role_name(self):
        collector = DictInputCollector({})
        with pytest.raises(ScenarioInputValidationError) as exc_info:
            collector.collect(role=_scalar_role(name="x"))
        assert exc_info.value.role_name == "x"

    def test_missing_optional_returns_default(self):
        collector = DictInputCollector({})
        role = _scalar_role(required=False, default="defaulted")
        assert collector.collect(role=role) == "defaulted"

    def test_defensive_copy_of_input_mapping(self):
        """Mutating the source dict after construction does not affect the collector."""
        source: dict[str, Any] = {"label": "first"}
        collector = DictInputCollector(source)
        source["label"] = "second"
        assert collector.collect(role=_scalar_role()) == "first"

    def test_collector_implements_protocol(self):
        assert isinstance(DictInputCollector({}), InputCollector)


# --- CliInputCollector -----------------------------------------------------------


class _ScriptedInput:
    """Helper: returns a series of canned responses, panics if exhausted."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError(f"Scripted input exhausted; got extra prompt {prompt!r}")
        return self._responses.pop(0)


class _Capture:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, msg: str) -> None:
        self.lines.append(msg)


class TestCliInputCollectorBasic:
    def test_scalar_string_value(self):
        scripted = _ScriptedInput(["harm"])
        collector = CliInputCollector(input_fn=scripted, output_fn=_Capture())
        assert collector.collect(role=_scalar_role()) == "harm"
        assert "label" in scripted.prompts[0]

    def test_scalar_int_coercion(self):
        scripted = _ScriptedInput(["42"])
        collector = CliInputCollector(input_fn=scripted, output_fn=_Capture())
        assert collector.collect(role=_scalar_role(param_type=int)) == 42

    def test_scalar_float_coercion(self):
        scripted = _ScriptedInput(["3.14"])
        collector = CliInputCollector(input_fn=scripted, output_fn=_Capture())
        assert collector.collect(role=_scalar_role(param_type=float)) == 3.14

    @pytest.mark.parametrize(
        "raw,expected", [("true", True), ("yes", True), ("1", True), ("false", False), ("n", False), ("0", False)]
    )
    def test_scalar_bool_coercion(self, raw, expected):
        scripted = _ScriptedInput([raw])
        collector = CliInputCollector(input_fn=scripted, output_fn=_Capture())
        assert collector.collect(role=_scalar_role(param_type=bool)) is expected

    def test_scalar_bool_invalid_raises_validation_error(self):
        scripted = _ScriptedInput(["maybe"])
        collector = CliInputCollector(input_fn=scripted, output_fn=_Capture())
        with pytest.raises(ScenarioInputValidationError) as exc_info:
            collector.collect(role=_scalar_role(param_type=bool))
        assert exc_info.value.role_name == "label"

    def test_scalar_int_invalid_raises_validation_error(self):
        scripted = _ScriptedInput(["not-a-number"])
        collector = CliInputCollector(input_fn=scripted, output_fn=_Capture())
        with pytest.raises(ScenarioInputValidationError):
            collector.collect(role=_scalar_role(param_type=int))


class TestCliInputCollectorChoice:
    def test_valid_choice(self):
        scripted = _ScriptedInput(["fast"])
        collector = CliInputCollector(input_fn=scripted, output_fn=_Capture())
        assert collector.collect(role=_choice_role()) == "fast"

    def test_invalid_choice_raises(self):
        scripted = _ScriptedInput(["bogus"])
        collector = CliInputCollector(input_fn=scripted, output_fn=_Capture())
        with pytest.raises(ScenarioInputValidationError) as exc_info:
            collector.collect(role=_choice_role())
        assert exc_info.value.role_name == "mode"

    def test_choice_prompt_lists_choices(self):
        scripted = _ScriptedInput(["fast"])
        capture = _Capture()
        collector = CliInputCollector(input_fn=scripted, output_fn=capture)
        collector.collect(role=_choice_role())
        assert "fast" in scripted.prompts[0]
        assert "slow" in scripted.prompts[0]


class TestCliInputCollectorOpaque:
    def test_opaque_role_raises_clear_error(self):
        collector = CliInputCollector(input_fn=_ScriptedInput([]), output_fn=_Capture())
        with pytest.raises(OpaqueRoleNotElicitableError) as exc_info:
            collector.collect(role=_opaque_role(name="atomic_attack"))
        assert exc_info.value.role_name == "atomic_attack"
        assert "from-artifact" in str(exc_info.value)


class TestCliInputCollectorOptional:
    def test_blank_response_uses_default(self):
        scripted = _ScriptedInput([""])
        collector = CliInputCollector(input_fn=scripted, output_fn=_Capture())
        role = _scalar_role(required=False, default="defaulted")
        assert collector.collect(role=role) == "defaulted"

    def test_blank_response_required_returns_empty_string(self):
        """Empty response on a required role falls through to coercion (or no coercion)."""
        scripted = _ScriptedInput([""])
        collector = CliInputCollector(input_fn=scripted, output_fn=_Capture())
        # str role, no coercion needed — empty string is returned as-is for required str roles.
        assert collector.collect(role=_scalar_role(param_type=str)) == ""


class TestCliInputCollectorErrorRecovery:
    def test_error_argument_renders_before_prompt(self):
        scripted = _ScriptedInput(["harm"])
        capture = _Capture()
        collector = CliInputCollector(input_fn=scripted, output_fn=capture)
        prior_error = ScenarioInputValidationError("bad value", role_name="label")
        collector.collect(role=_scalar_role(), error=prior_error, attempt=1)
        assert any("Previous attempt failed" in line for line in capture.lines)
        assert any("bad value" in line for line in capture.lines)

    def test_attempt_counter_shows_in_prompt(self):
        scripted = _ScriptedInput(["x"])
        collector = CliInputCollector(input_fn=scripted, output_fn=_Capture())
        collector.collect(role=_scalar_role(), attempt=2)
        assert "attempt 3" in scripted.prompts[0]


# --- ArtifactInputCollector ------------------------------------------------------


class TestArtifactInputCollector:
    def test_replays_value(self):
        collector = ArtifactInputCollector({"label": "from-artifact"})
        assert collector.collect(role=_scalar_role()) == "from-artifact"

    def test_missing_required_raises(self):
        collector = ArtifactInputCollector({})
        with pytest.raises(ScenarioInputValidationError) as exc_info:
            collector.collect(role=_scalar_role())
        assert exc_info.value.role_name == "label"

    def test_missing_optional_returns_default(self):
        collector = ArtifactInputCollector({})
        role = _scalar_role(required=False, default="d")
        assert collector.collect(role=role) == "d"

    def test_opaque_payload_passes_through(self):
        """Opaque values stored in the artifact (as ``ComponentIdentifier.to_dict()``) replay verbatim."""
        opaque_payload = {"class_name": "AtomicAttack", "init_data": {}}
        collector = ArtifactInputCollector({"instance": opaque_payload})
        assert collector.collect(role=_opaque_role()) == opaque_payload

    def test_defensive_copy_of_input_mapping(self):
        """Mutating the source dict after construction does not affect the collector."""
        source: dict[str, Any] = {"label": "first"}
        collector = ArtifactInputCollector(source)
        source["label"] = "second"
        assert collector.collect(role=_scalar_role()) == "first"

    def test_implements_input_collector_protocol(self):
        assert isinstance(ArtifactInputCollector({}), InputCollector)


class TestInputCollectorProtocolNegative:
    """Pin that ``@runtime_checkable`` does not over-match objects without ``collect``."""

    def test_str_is_not_collector(self):
        assert not isinstance("not a collector", InputCollector)

    def test_dict_is_not_collector(self):
        assert not isinstance({"label": "v"}, InputCollector)

    def test_object_with_unrelated_attrs_is_not_collector(self):
        class _NotACollector:
            def some_other_method(self) -> None:
                pass

        assert not isinstance(_NotACollector(), InputCollector)

    def test_object_with_collect_attr_is_collector(self):
        """``@runtime_checkable`` only checks attribute presence, not signature."""

        class _DuckTyped:
            def collect(self, *, role: Any, error: Any = None, attempt: int = 0) -> Any:
                return "ok"

        assert isinstance(_DuckTyped(), InputCollector)


# --- collect_inputs_with_retry ---------------------------------------------------


class _FlakyCollector:
    """Collector that fails the first ``fail_n`` attempts, then succeeds."""

    def __init__(self, *, fail_n: int, success_value: Any) -> None:
        self.fail_n = fail_n
        self.success_value = success_value
        self.attempts_seen: list[int] = []
        self.errors_seen: list[Exception | None] = []

    def collect(self, *, role: RoleDescriptor, error: Exception | None = None, attempt: int = 0) -> Any:
        self.attempts_seen.append(attempt)
        self.errors_seen.append(error)
        if attempt < self.fail_n:
            raise ScenarioInputValidationError(f"attempt {attempt} failed", role_name=role.name)
        return self.success_value


class TestCollectInputsWithRetrySuccess:
    def test_single_role_first_try(self):
        collector = DictInputCollector({"label": "v"})
        result = collect_inputs_with_retry(collector=collector, schema=[_scalar_role()])
        assert result == {"label": "v"}

    def test_multiple_roles_in_declared_order(self):
        schema = [_scalar_role(name="a"), _scalar_role(name="b"), _scalar_role(name="c")]
        collector = DictInputCollector({"a": 1, "b": 2, "c": 3})
        result = collect_inputs_with_retry(collector=collector, schema=schema)
        assert list(result.keys()) == ["a", "b", "c"]

    def test_optional_missing_role_omitted_from_result(self):
        """If a collector returns the default, the role is still in the result map."""
        schema = [_scalar_role(name="a", required=False, default="d")]
        collector = DictInputCollector({})
        result = collect_inputs_with_retry(collector=collector, schema=schema)
        assert result == {"a": "d"}


class TestCollectInputsWithRetryRetries:
    def test_succeeds_on_third_attempt(self):
        collector = _FlakyCollector(fail_n=2, success_value="ok")
        result = collect_inputs_with_retry(collector=collector, schema=[_scalar_role()], max_attempts=5)
        assert result == {"label": "ok"}
        assert collector.attempts_seen == [0, 1, 2]

    def test_error_propagates_to_next_attempt(self):
        collector = _FlakyCollector(fail_n=1, success_value="ok")
        collect_inputs_with_retry(collector=collector, schema=[_scalar_role()], max_attempts=5)
        assert collector.errors_seen[0] is None
        assert isinstance(collector.errors_seen[1], ScenarioInputValidationError)
        assert collector.errors_seen[1].role_name == "label"  # type: ignore[union-attr]

    def test_exhausts_attempts_and_raises(self):
        collector = _FlakyCollector(fail_n=10, success_value="never")
        with pytest.raises(MaxAttemptsExceededError) as exc_info:
            collect_inputs_with_retry(collector=collector, schema=[_scalar_role()], max_attempts=3)
        assert exc_info.value.role_name == "label"
        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_error, ScenarioInputValidationError)
        # Tried exactly max_attempts times (no extra retry past the budget).
        assert collector.attempts_seen == [0, 1, 2]


class TestOpaqueRoleNotElicitableError:
    def test_message_points_at_artifact_path(self):
        exc = OpaqueRoleNotElicitableError("atomic_attack")
        assert "atomic_attack" in str(exc)
        assert "from-artifact" in str(exc)

    def test_is_not_implemented_error(self):
        """Allows callers to use the broader ``NotImplementedError`` catch."""
        assert issubclass(OpaqueRoleNotElicitableError, NotImplementedError)
