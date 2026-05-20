# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Phase 8c — coverage for the scenario builder module functions."""

from __future__ import annotations

from typing import Any, cast

import pytest

from pyrit.common.parameter import Parameter
from pyrit.scenario.core.builder import (
    ScenarioInputValidationError,
    build_scenario_from_inputs,
    discover_input_schema,
    discover_supported_parameters,
    validate_init_async_inputs,
    validate_init_inputs,
)
from pyrit.scenario.core.input_schema import RoleDescriptor, RoleTag


class _FakeScenarioBase:
    """Minimal duck-typed stand-in for ``Scenario`` to exercise the builder.

    The builder only touches ``input_schema`` and ``supported_parameters`` at
    discovery time and ``__init__`` / ``initialize_async`` at build time, so we
    deliberately avoid the real ``Scenario`` heavyweight constructor (memory,
    identifier, deprecation machinery).
    """

    _schema: list[RoleDescriptor] = []
    _params: list[Parameter] = []

    @classmethod
    def input_schema(cls) -> list[RoleDescriptor]:
        return cls._schema

    @classmethod
    def supported_parameters(cls) -> list[Parameter]:
        return cls._params


class _FakeScenarioNoArgs(_FakeScenarioBase):
    def __init__(self) -> None:
        self.init_called = True
        self.init_async_called = False

    async def initialize_async(self) -> None:
        self.init_async_called = True


class _FakeScenarioScalarRoles(_FakeScenarioBase):
    _schema = [
        RoleDescriptor(name="weakness_label", description="Label", tag=RoleTag.SCALAR, param_type=str),
        RoleDescriptor(
            name="threshold",
            description="Score cutoff",
            tag=RoleTag.SCALAR,
            param_type=float,
            default=0.5,
            required=False,
        ),
    ]
    _params = [
        Parameter(name="max_concurrency", description="Concurrency", default=1, param_type=int),
    ]

    def __init__(self, *, weakness_label: str, threshold: float = 0.5) -> None:
        self.weakness_label = weakness_label
        self.threshold = threshold
        self.init_async_max_concurrency: int | None = None

    async def initialize_async(self, *, max_concurrency: int = 1) -> None:
        self.init_async_max_concurrency = max_concurrency


class _FakeScenarioChoice(_FakeScenarioBase):
    _schema = [
        RoleDescriptor(
            name="mode",
            description="Pick a mode",
            tag=RoleTag.CHOICE,
            param_type=str,
            choices=("fast", "thorough"),
        ),
    ]

    def __init__(self, *, mode: str) -> None:
        self.mode = mode

    async def initialize_async(self) -> None:
        pass


class _FakeScenarioRaises(_FakeScenarioBase):
    _schema = [
        RoleDescriptor(name="value", description="anything", tag=RoleTag.SCALAR, param_type=int),
    ]

    def __init__(self, *, value: int) -> None:
        if value < 0:
            raise ValueError(f"value must be non-negative; got {value}")
        self.value = value

    async def initialize_async(self) -> None:
        pass


class _FakeScenarioInitAsyncRaises(_FakeScenarioBase):
    def __init__(self) -> None:
        pass

    async def initialize_async(self) -> None:
        raise RuntimeError("initialize_async failed")


class TestDiscoverInputSchema:
    def test_returns_list_copy(self):
        schema = discover_input_schema(cast("Any", _FakeScenarioScalarRoles))
        assert isinstance(schema, list)
        assert len(schema) == 2
        assert schema[0].name == "weakness_label"

    def test_empty_schema_default(self):
        schema = discover_input_schema(cast("Any", _FakeScenarioNoArgs))
        assert schema == []

    def test_discover_does_not_share_mutable_list(self):
        """Mutating the returned list does not affect subsequent calls."""
        schema_a = discover_input_schema(cast("Any", _FakeScenarioScalarRoles))
        schema_a.clear()
        schema_b = discover_input_schema(cast("Any", _FakeScenarioScalarRoles))
        assert len(schema_b) == 2


class TestDiscoverSupportedParameters:
    def test_returns_list(self):
        params = discover_supported_parameters(cast("Any", _FakeScenarioScalarRoles))
        assert isinstance(params, list)
        assert len(params) == 1
        assert params[0].name == "max_concurrency"

    def test_empty_when_unset(self):
        params = discover_supported_parameters(cast("Any", _FakeScenarioNoArgs))
        assert params == []


class TestValidateInitInputs:
    def test_all_required_present_passes(self):
        validate_init_inputs(schema=_FakeScenarioScalarRoles._schema, init_inputs={"weakness_label": "harm"})

    def test_missing_required_raises_with_role_name(self):
        with pytest.raises(ScenarioInputValidationError) as exc_info:
            validate_init_inputs(schema=_FakeScenarioScalarRoles._schema, init_inputs={})
        assert exc_info.value.role_name == "weakness_label"
        assert "weakness_label" in str(exc_info.value)

    def test_missing_optional_passes(self):
        """Optional role absence is not a validation failure."""
        validate_init_inputs(schema=_FakeScenarioScalarRoles._schema, init_inputs={"weakness_label": "x"})

    def test_choice_value_in_choices_passes(self):
        validate_init_inputs(schema=_FakeScenarioChoice._schema, init_inputs={"mode": "fast"})

    def test_choice_value_not_in_choices_raises(self):
        with pytest.raises(ScenarioInputValidationError) as exc_info:
            validate_init_inputs(schema=_FakeScenarioChoice._schema, init_inputs={"mode": "instant"})
        assert exc_info.value.role_name == "mode"
        assert "instant" in str(exc_info.value)

    def test_unknown_keys_pass_through_silently(self):
        """Scenarios may accept kwargs not in the schema (e.g. scenario_result_id)."""
        validate_init_inputs(
            schema=_FakeScenarioScalarRoles._schema,
            init_inputs={"weakness_label": "x", "scenario_result_id": "abc"},
        )

    def test_empty_schema_accepts_any_inputs(self):
        validate_init_inputs(schema=[], init_inputs={"whatever": 1})


class TestValidateInitAsyncInputs:
    def test_accepts_all_known_keys(self):
        validate_init_async_inputs(
            scenario_cls=cast("Any", _FakeScenarioScalarRoles),
            init_async_inputs={"max_concurrency": 4},
        )

    def test_accepts_empty(self):
        validate_init_async_inputs(
            scenario_cls=cast("Any", _FakeScenarioScalarRoles),
            init_async_inputs={},
        )

    def test_rejects_unknown_key(self):
        with pytest.raises(ScenarioInputValidationError) as exc_info:
            validate_init_async_inputs(
                scenario_cls=cast("Any", _FakeScenarioScalarRoles),
                init_async_inputs={"typo": 1},
            )
        message = str(exc_info.value)
        assert "typo" in message
        # The error should also list what *is* accepted to help the user recover.
        assert "max_concurrency" in message

    def test_rejects_multiple_unknown_keys(self):
        with pytest.raises(ScenarioInputValidationError) as exc_info:
            validate_init_async_inputs(
                scenario_cls=cast("Any", _FakeScenarioScalarRoles),
                init_async_inputs={"foo": 1, "bar": 2},
            )
        message = str(exc_info.value)
        assert "foo" in message and "bar" in message

    def test_var_keyword_opts_out(self):
        """Scenarios whose ``initialize_async`` accepts ``**kwargs`` skip the check."""

        class _VarKw(_FakeScenarioBase):
            async def initialize_async(self, **kwargs: Any) -> None:
                pass

        # Should not raise even with an arbitrary key.
        validate_init_async_inputs(
            scenario_cls=cast("Any", _VarKw),
            init_async_inputs={"anything": "goes"},
        )


class TestBuildScenarioFromInputs:
    async def test_constructs_and_initializes(self):
        scenario = await build_scenario_from_inputs(
            cast("Any", _FakeScenarioScalarRoles),
            init_inputs={"weakness_label": "harm"},
            init_async_inputs={"max_concurrency": 4},
        )
        assert scenario.weakness_label == "harm"  # type: ignore[attr-defined]
        assert scenario.threshold == 0.5  # type: ignore[attr-defined]
        assert scenario.init_async_max_concurrency == 4  # type: ignore[attr-defined]

    async def test_no_args_scenario(self):
        scenario = await build_scenario_from_inputs(
            cast("Any", _FakeScenarioNoArgs),
            init_inputs={},
            init_async_inputs={},
        )
        assert scenario.init_async_called is True  # type: ignore[attr-defined]

    async def test_validation_runs_before_construction(self):
        """A missing required role raises before ``__init__`` is reached."""
        with pytest.raises(ScenarioInputValidationError) as exc_info:
            await build_scenario_from_inputs(
                cast("Any", _FakeScenarioScalarRoles),
                init_inputs={},
                init_async_inputs={},
            )
        assert exc_info.value.role_name == "weakness_label"

    async def test_construction_errors_propagate(self):
        """A ``__init__`` exception is not wrapped — caller gets the original."""
        with pytest.raises(ValueError, match="value must be non-negative"):
            await build_scenario_from_inputs(
                cast("Any", _FakeScenarioRaises),
                init_inputs={"value": -1},
                init_async_inputs={},
            )

    async def test_initialize_async_errors_propagate(self):
        with pytest.raises(RuntimeError, match="initialize_async failed"):
            await build_scenario_from_inputs(
                cast("Any", _FakeScenarioInitAsyncRaises),
                init_inputs={},
                init_async_inputs={},
            )

    async def test_choice_validation_fires(self):
        with pytest.raises(ScenarioInputValidationError) as exc_info:
            await build_scenario_from_inputs(
                cast("Any", _FakeScenarioChoice),
                init_inputs={"mode": "bogus"},
                init_async_inputs={},
            )
        assert exc_info.value.role_name == "mode"

    async def test_init_async_inputs_unknown_key_raises_validation_error(self):
        """Lead 2: an unknown init_async_inputs key surfaces as ScenarioInputValidationError.

        Without pre-validation, ``initialize_async(**init_async_inputs)`` blows up with
        a raw ``TypeError`` from Python's call machinery (via the @apply_defaults
        wrapper's ``sig.bind``). The wizard's retry loop catches only
        ``ScenarioInputValidationError``, so a typo'd flag would crash the wizard
        instead of surfacing as a recoverable validation error.
        """
        with pytest.raises(ScenarioInputValidationError) as exc_info:
            await build_scenario_from_inputs(
                cast("Any", _FakeScenarioScalarRoles),
                init_inputs={"weakness_label": "harm"},
                init_async_inputs={"max_concurrency": 4, "bogus_typo": "value"},
            )
        assert "bogus_typo" in str(exc_info.value)

    async def test_init_async_inputs_multiple_unknown_keys_all_listed(self):
        with pytest.raises(ScenarioInputValidationError) as exc_info:
            await build_scenario_from_inputs(
                cast("Any", _FakeScenarioScalarRoles),
                init_inputs={"weakness_label": "harm"},
                init_async_inputs={"alpha": 1, "beta": 2},
            )
        message = str(exc_info.value)
        assert "alpha" in message and "beta" in message

    async def test_init_async_inputs_unknown_key_does_not_construct_scenario(self):
        """Validation must run before ``__init__`` to avoid orphaned construction side effects."""

        class _TracksConstruction(_FakeScenarioBase):
            constructed = False

            def __init__(self) -> None:
                type(self).constructed = True

            async def initialize_async(self) -> None:
                pass

        with pytest.raises(ScenarioInputValidationError):
            await build_scenario_from_inputs(
                cast("Any", _TracksConstruction),
                init_inputs={},
                init_async_inputs={"bogus": 1},
            )
        assert _TracksConstruction.constructed is False

    async def test_init_async_inputs_var_keyword_accepts_anything(self):
        """A scenario whose ``initialize_async`` accepts ``**kwargs`` opts out of validation."""

        class _VarKw(_FakeScenarioBase):
            def __init__(self) -> None:
                pass

            async def initialize_async(self, **kwargs: Any) -> None:
                self.received_kwargs = kwargs

        scenario = await build_scenario_from_inputs(
            cast("Any", _VarKw),
            init_inputs={},
            init_async_inputs={"whatever": "ok"},
        )
        assert scenario.received_kwargs == {"whatever": "ok"}  # type: ignore[attr-defined]


class TestScenarioInputValidationError:
    def test_is_value_error_subclass(self):
        """Lets callers catch with the broader ``ValueError`` if they want."""
        assert issubclass(ScenarioInputValidationError, ValueError)

    def test_role_name_defaults_to_none(self):
        exc = ScenarioInputValidationError("bare message")
        assert exc.role_name is None

    def test_role_name_round_trips(self):
        exc = ScenarioInputValidationError("oops", role_name="x")
        assert exc.role_name == "x"
        assert str(exc) == "oops"
