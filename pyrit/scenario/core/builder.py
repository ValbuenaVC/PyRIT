# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario builder — module functions for constructing scenarios from declared inputs.

The builder is the single entry point used by the wizard CLI (Phase 8d), the
artifact loader (Phase 8g), and the inverse-waterfall ``pyrit_scan --from-artifact``
path (Phase 8e). It validates declared inputs against the scenario's
:meth:`Scenario.input_schema` (rich-object ``__init__`` roles) and
:meth:`Scenario.supported_parameters` (scalar ``initialize_async`` args), then
runs both lifecycle phases and returns a runnable scenario.

The retry-on-ValidationError loop is intentionally factored out into
:mod:`pyrit.scenario.core.input_collector` so the builder itself stays
collector-agnostic; the CLI driver wires the two together.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from pyrit.scenario.core.input_schema import RoleDescriptor, RoleTag

if TYPE_CHECKING:
    from pyrit.common.parameter import Parameter
    from pyrit.scenario.core.scenario import Scenario


class ScenarioInputValidationError(ValueError):
    """
    Raised when supplied init inputs fail validation against ``input_schema()``.

    Carries the role name (when applicable) so the collector retry loop can
    re-prompt the specific role rather than restarting collection from scratch.
    """

    def __init__(self, message: str, *, role_name: str | None = None) -> None:
        """Store ``message`` and remember which role failed (when known)."""
        super().__init__(message)
        self.role_name = role_name


def discover_input_schema(scenario_cls: type[Scenario]) -> list[RoleDescriptor]:
    """
    Return the declared rich-object ``__init__`` roles for the scenario.

    Thin wrapper around :meth:`Scenario.input_schema` so the wizard layer can
    treat schema discovery uniformly across all scenarios — including base-class
    scenarios that take the default ``[]`` from :class:`Scenario`.

    Args:
        scenario_cls: Concrete subclass of :class:`Scenario`.

    Returns:
        list[RoleDescriptor]: The role list returned by the class.
    """
    return list(scenario_cls.input_schema())


def discover_supported_parameters(scenario_cls: type[Scenario]) -> list[Parameter]:
    """
    Return the declared scalar ``initialize_async`` parameters for the scenario.

    Thin wrapper around :meth:`Scenario.supported_parameters` so the wizard layer
    can treat both halves of the lifecycle symmetrically.

    Args:
        scenario_cls: Concrete subclass of :class:`Scenario`.

    Returns:
        list[Parameter]: The parameter list returned by the class.
    """
    return list(scenario_cls.supported_parameters())


def validate_init_inputs(
    *,
    schema: list[RoleDescriptor],
    init_inputs: dict[str, Any],
) -> None:
    """
    Validate ``init_inputs`` against the scenario's declared ``input_schema``.

    The builder calls this before constructing the scenario so a malformed input
    set surfaces as a structured :class:`ScenarioInputValidationError` (with the
    offending role name) rather than a downstream constructor ``TypeError``.

    Checks performed:

    * Every required role is present in ``init_inputs``.
    * No ``CHOICE`` role's value falls outside its declared choices.

    Type coercion is deliberately left to the collector layer: CLI collectors
    coerce strings to typed primitives at elicitation time; programmatic
    collectors pass typed values directly; opaque roles bypass coercion entirely.

    Args:
        schema: The declared role list (typically from
            :func:`discover_input_schema`).
        init_inputs: Caller-supplied init inputs keyed by role name.

    Raises:
        ScenarioInputValidationError: If a required role is missing or a
            choice role's value is not in the declared choices.
    """
    schema_by_name = {role.name: role for role in schema}

    for role in schema:
        if role.required and role.name not in init_inputs:
            raise ScenarioInputValidationError(
                f"Missing required init input {role.name!r} ({role.tag.value}). Description: {role.description}",
                role_name=role.name,
            )

    for name, value in init_inputs.items():
        role = schema_by_name.get(name)
        if role is None:
            # Unknown keys are not an error: scenarios may accept kwargs that aren't
            # explicitly schema-declared (e.g. ``scenario_result_id``). Skip silently.
            continue

        if role.tag is RoleTag.CHOICE:
            assert role.choices is not None  # __post_init__ guarantees this
            if value not in role.choices:
                raise ScenarioInputValidationError(
                    f"Invalid value for role {name!r}: {value!r} is not in declared choices {role.choices!r}.",
                    role_name=name,
                )


def validate_init_async_inputs(
    *,
    scenario_cls: type[Scenario],
    init_async_inputs: dict[str, Any],
) -> None:
    """
    Validate ``init_async_inputs`` against the scenario's ``initialize_async`` signature.

    Catches unknown keyword arguments at the wizard layer before they surface as a
    raw ``TypeError`` from Python's call machinery (the ``@apply_defaults`` wrapper
    around ``initialize_async`` calls ``inspect.Signature.bind`` and lets a
    ``TypeError`` propagate for unknown kwargs). The wizard's retry loop in
    :func:`pyrit.scenario.core.input_collector.collect_inputs_with_retry` catches
    only :class:`ScenarioInputValidationError`, so an unwrapped ``TypeError`` would
    crash the wizard rather than re-prompt.

    Type coercion is deliberately not performed here — the collector layer is
    responsible for upstream coercion of scalars. Scenarios whose
    ``initialize_async`` accepts ``**kwargs`` opt out of unknown-key validation.

    Args:
        scenario_cls: Concrete subclass of :class:`Scenario`.
        init_async_inputs: Caller-supplied ``initialize_async`` arguments.

    Raises:
        ScenarioInputValidationError: If ``init_async_inputs`` contains a keyword
            argument that ``initialize_async`` does not accept.
    """
    sig = inspect.signature(scenario_cls.initialize_async)
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_kw:
        return

    accepted = {
        name
        for name, param in sig.parameters.items()
        if name != "self" and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unknown = sorted(set(init_async_inputs) - accepted)
    if unknown:
        raise ScenarioInputValidationError(
            f"Unknown init_async_inputs keys for {scenario_cls.__name__}: {unknown!r}. "
            f"Accepted keys: {sorted(accepted)!r}."
        )


async def build_scenario_from_inputs(
    scenario_cls: type[Scenario],
    *,
    init_inputs: dict[str, Any],
    init_async_inputs: dict[str, Any],
) -> Scenario:
    """
    Construct a scenario, run ``initialize_async``, and return the runnable instance.

    This is the single entry point used by the wizard CLI (Phase 8d) and the
    artifact loader (Phase 8g). It validates ``init_inputs`` against the
    scenario's ``input_schema`` first so a malformed input set surfaces as
    :class:`ScenarioInputValidationError` (carrying the offending role name)
    rather than a deep ``TypeError`` from the constructor.

    ``init_async_inputs`` are passed verbatim to ``initialize_async``. The
    existing :meth:`Scenario.set_params_from_args` machinery validates
    scenario-declared parameters; here we additionally pre-check that every key
    in ``init_async_inputs`` is accepted by ``initialize_async``'s signature, so
    typo'd keys surface as :class:`ScenarioInputValidationError` (recoverable by
    the wizard's retry loop) rather than a raw ``TypeError`` (which would crash
    the loop). Scenarios whose ``initialize_async`` accepts ``**kwargs`` opt out
    of the unknown-key check.

    Args:
        scenario_cls: Concrete subclass of :class:`Scenario`.
        init_inputs: Rich-object ``__init__`` arguments keyed by role name. All
            required roles from ``input_schema()`` must be present.
        init_async_inputs: Scalar ``initialize_async`` arguments. Validated
            against the ``initialize_async`` signature for unknown keys, and by
            ``Scenario.set_params_from_args`` via ``supported_parameters()``.

    Returns:
        Scenario: An initialized, runnable scenario instance.

    Raises:
        ScenarioInputValidationError: If ``init_inputs`` fails validation or
            ``init_async_inputs`` contains keys not accepted by
            ``initialize_async``.
    """
    schema = discover_input_schema(scenario_cls)
    validate_init_inputs(schema=schema, init_inputs=init_inputs)
    validate_init_async_inputs(scenario_cls=scenario_cls, init_async_inputs=init_async_inputs)

    scenario = scenario_cls(**init_inputs)
    await scenario.initialize_async(**init_async_inputs)
    return scenario
