# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Input collectors — elicit ``init_inputs`` from various sources for the wizard.

The :class:`InputCollector` Protocol is the front-end abstraction: a CLI
implementation reads from stdin, a programmatic one reads from a dict, and an
artifact-replay one reads from a previously saved
:class:`pyrit.scenario.core.graph_artifact.GraphArtifact`. The
:func:`collect_inputs_with_retry` helper drives the error-recovery loop with
the builder's :class:`pyrit.scenario.core.builder.ScenarioInputValidationError`.

Opaque roles (pre-built ``Identifiable`` instances) cannot be elicited from a
pure CLI flow — the :class:`CliInputCollector` reference impl raises
:class:`OpaqueRoleNotElicitableError` with a pointer at ``--from-artifact``
when it encounters one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pyrit.scenario.core.builder import ScenarioInputValidationError
from pyrit.scenario.core.input_schema import RoleDescriptor, RoleTag

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class OpaqueRoleNotElicitableError(NotImplementedError):
    """Raised when a CLI collector encounters an OPAQUE role it cannot elicit."""

    def __init__(self, role_name: str) -> None:
        """Format a help message pointing at the artifact-replay workaround."""
        super().__init__(
            f"Role {role_name!r} is OPAQUE (a pre-built Identifiable instance) and cannot "
            "be elicited from a CLI prompt. Provide it programmatically via "
            "``build_scenario_from_inputs(..., init_inputs={...})`` or replay it from a "
            "saved graph artifact via ``pyrit_scan --from-artifact path.yaml``."
        )
        self.role_name = role_name


class MaxAttemptsExceededError(RuntimeError):
    """Raised when a collector loop exceeds the per-role retry budget."""

    def __init__(self, role_name: str, attempts: int, last_error: Exception | None) -> None:
        """Format a diagnostic message including the last seen validation error."""
        message = (
            f"Exceeded {attempts} elicitation attempts for role {role_name!r}. Last error: {last_error!r}"
            if last_error
            else f"Exceeded {attempts} elicitation attempts for role {role_name!r}."
        )
        super().__init__(message)
        self.role_name = role_name
        self.attempts = attempts
        self.last_error = last_error


@runtime_checkable
class InputCollector(Protocol):
    """
    Protocol for eliciting a single role's value.

    Implementations may be sync (return immediately) or wrap async work; the
    wizard driver awaits each ``collect`` call so both forms are supported.
    The ``error`` and ``attempt`` arguments enable the retry loop in
    :func:`collect_inputs_with_retry`: on a validation failure the driver
    re-invokes ``collect`` with the prior exception and an incremented attempt
    counter, so collectors can re-prompt with the error context visible to the
    user.
    """

    def collect(
        self,
        *,
        role: RoleDescriptor,
        error: Exception | None = None,
        attempt: int = 0,
    ) -> Any:
        """Return the value for ``role`` (see implementations for source-specific behavior)."""
        ...


class DictInputCollector:
    """
    Programmatic / test collector that returns values from a supplied dict.

    Mainly used in unit tests and by the wizard library API. Missing required
    roles raise :class:`ScenarioInputValidationError` (consistent with the
    builder's surface) so the test author sees the same error a CLI user would.
    """

    def __init__(self, values: Mapping[str, Any]) -> None:
        """
        Initialize the collector.

        Args:
            values: Mapping from role name to value. Roles absent from the
                mapping return their declared default if optional, or raise
                :class:`ScenarioInputValidationError` if required.
        """
        self._values = dict(values)

    def collect(
        self,
        *,
        role: RoleDescriptor,
        error: Exception | None = None,
        attempt: int = 0,
    ) -> Any:
        """
        Return the value for ``role`` from the supplied dict.

        Returns:
            Any: The value bound to ``role.name`` in the dict, or ``role.default``
                if the role is optional and absent.

        Raises:
            ScenarioInputValidationError: If the role is required and absent from
                the dict.
        """
        if role.name in self._values:
            return self._values[role.name]
        if role.required:
            raise ScenarioInputValidationError(
                f"DictInputCollector has no value for required role {role.name!r}.",
                role_name=role.name,
            )
        return role.default


class CliInputCollector:
    """
    Stdin/stdout reference collector for interactive elicitation.

    Reads one role at a time via the standard :func:`input` builtin (overridable
    via ``input_fn`` for testability). Coerces strings to the role's declared
    ``param_type`` for ``SCALAR`` roles and validates membership for ``CHOICE``
    roles. Refuses to elicit ``OPAQUE`` roles (see
    :class:`OpaqueRoleNotElicitableError`).

    Error-recovery contract: when invoked with a non-None ``error``, the
    collector prefixes the re-prompt with the error message so the user sees
    what went wrong before re-entering.
    """

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] | None = None,
        output_fn: Callable[[str], None] | None = None,
    ) -> None:
        """
        Initialize the CLI collector.

        Args:
            input_fn: Function used to read a line from the user. Defaults to
                the builtin :func:`input`. Override for testing.
            output_fn: Function used to write a prompt line. Defaults to
                :func:`print`. Override for testing.
        """
        self._input = input_fn or input
        self._print = output_fn or (lambda msg: print(msg))

    def collect(
        self,
        *,
        role: RoleDescriptor,
        error: Exception | None = None,
        attempt: int = 0,
    ) -> Any:
        """
        Prompt the user for the role's value and return it (coerced if applicable).

        Args:
            role: The role to elicit.
            error: Previous validation error, if any, surfaced to the user.
            attempt: Current attempt count (0 = first prompt).

        Returns:
            The collected (and coerced) value.

        Raises:
            OpaqueRoleNotElicitableError: If ``role.tag is RoleTag.OPAQUE``.
            ScenarioInputValidationError: If the user enters a value that
                fails coercion or choice-membership validation.
        """
        if role.tag is RoleTag.OPAQUE:
            raise OpaqueRoleNotElicitableError(role.name)

        if error is not None:
            self._print(f"  ! Previous attempt failed: {error}")

        prompt = self._render_prompt(role=role, attempt=attempt)
        raw = self._input(prompt).strip()

        if raw == "" and not role.required:
            return role.default

        if role.tag is RoleTag.CHOICE:
            return self._validate_choice(role=role, raw=raw)

        if role.tag is RoleTag.SCALAR and role.param_type is not None:
            return self._coerce_scalar(role=role, raw=raw)

        # SCALAR with no param_type, REGISTRY_REF, FACTORY — return string as-is.
        # The builder / downstream code is responsible for any deeper coercion.
        return raw

    def _render_prompt(self, *, role: RoleDescriptor, attempt: int) -> str:
        bits = [f"{role.name}"]
        if role.description:
            bits.append(f" ({role.description})")
        if role.tag is RoleTag.CHOICE and role.choices is not None:
            bits.append(f" [choices: {', '.join(map(str, role.choices))}]")
        if not role.required and role.default is not None:
            bits.append(f" [default: {role.default!r}]")
        elif not role.required:
            bits.append(" [optional]")
        prefix = f"(attempt {attempt + 1}) " if attempt > 0 else ""
        return f"{prefix}{''.join(bits)}: "

    def _validate_choice(self, *, role: RoleDescriptor, raw: str) -> Any:
        assert role.choices is not None
        # Compare raw string against str(choice); accept the original-typed choice on match.
        for choice in role.choices:
            if str(choice) == raw or choice == raw:
                return choice
        raise ScenarioInputValidationError(
            f"Invalid choice for role {role.name!r}: {raw!r} not in {role.choices!r}.",
            role_name=role.name,
        )

    def _coerce_scalar(self, *, role: RoleDescriptor, raw: str) -> Any:
        target = role.param_type
        try:
            if target is bool:
                if raw.lower() in {"true", "yes", "y", "1"}:
                    return True
                if raw.lower() in {"false", "no", "n", "0"}:
                    return False
                raise ValueError(f"Cannot interpret {raw!r} as bool.")
            if target is int:
                return int(raw)
            if target is float:
                return float(raw)
            if target is str:
                return raw
        except (TypeError, ValueError) as exc:
            raise ScenarioInputValidationError(
                f"Failed to coerce {raw!r} to {target!r} for role {role.name!r}: {exc}",
                role_name=role.name,
            ) from exc
        # Unrecognized param_type — pass through as string.
        return raw


class ArtifactInputCollector:
    """
    Collector that replays init_inputs from a previously saved graph artifact.

    Used by ``pyrit_scan --from-artifact`` (Phase 8e) and by the artifact
    loader (Phase 8g). Looks each role up by name in the artifact's
    ``init_inputs`` dict. Opaque roles surface as their stored
    ``ComponentIdentifier.to_dict()`` payload — the load path is responsible
    for re-materializing them into live instances.
    """

    def __init__(self, init_inputs: Mapping[str, Any]) -> None:
        """
        Initialize the collector.

        Args:
            init_inputs: The ``init_inputs`` map from a saved
                :class:`pyrit.scenario.core.graph_artifact.GraphArtifact`.
        """
        self._init_inputs = dict(init_inputs)

    def collect(
        self,
        *,
        role: RoleDescriptor,
        error: Exception | None = None,
        attempt: int = 0,
    ) -> Any:
        """
        Return the recorded value for ``role`` from the artifact.

        Returns:
            Any: The value recorded under ``role.name`` in the artifact, or
                ``role.default`` if the role is optional and absent.

        Raises:
            ScenarioInputValidationError: If the role is required and absent from
                the artifact.
        """
        if role.name in self._init_inputs:
            return self._init_inputs[role.name]
        if role.required:
            raise ScenarioInputValidationError(
                f"Artifact has no recorded value for required role {role.name!r}.",
                role_name=role.name,
            )
        return role.default


def collect_inputs_with_retry(
    *,
    collector: InputCollector,
    schema: list[RoleDescriptor],
    max_attempts: int = 5,
) -> dict[str, Any]:
    """
    Drive an :class:`InputCollector` to gather a full init-input set.

    Iterates the schema in declared order. For each role, calls
    :meth:`InputCollector.collect`. If the call raises
    :class:`ScenarioInputValidationError`, the loop retries up to
    ``max_attempts`` times with the prior error passed back to the collector
    so it can re-prompt with context. After exhausting the budget, raises
    :class:`MaxAttemptsExceededError`.

    Args:
        collector: Any :class:`InputCollector` implementation.
        schema: The schema to collect against (typically from
            :func:`pyrit.scenario.core.builder.discover_input_schema`).
        max_attempts: Maximum attempts per role before giving up. Defaults to 5.

    Returns:
        dict[str, Any]: Collected values keyed by role name. Optional roles
            absent from the collector are omitted (the scenario constructor
            then uses its own default).

    Raises:
        MaxAttemptsExceededError: When the per-role retry budget is exhausted.
    """
    collected: dict[str, Any] = {}
    for role in schema:
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                value = collector.collect(role=role, error=last_error, attempt=attempt)
            except ScenarioInputValidationError as exc:
                last_error = exc
                continue
            collected[role.name] = value
            break
        else:
            raise MaxAttemptsExceededError(role.name, max_attempts, last_error)
    return collected
