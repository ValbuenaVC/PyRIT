# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
RoleDescriptor — declarative schema for a scenario's ``__init__`` inputs.

A scenario's ``__init__`` arguments fall into a small taxonomy:

* **scalar**       — primitive values (``int``, ``str``, ``bool``, ``float``).
* **choice**       — a fixed set of literal alternatives.
* **registry_ref** — a name into a PyRIT registry (e.g. a target name).
* **factory**      — a structured spec elicited via nested schema (``AttackTechniqueSpec``).
* **opaque**       — a pre-built ``Identifiable`` instance the CLI wizard cannot
                     elicit (``AtomicAttack``, ``OutcomeScorer``, etc.). Programmatic
                     callers can supply these directly; CLI flows must use an artifact.

``Scenario.input_schema()`` returns ``list[RoleDescriptor]`` declaring the rich-object
``__init__`` inputs; ``Scenario.supported_parameters()`` already declares the scalar
``initialize_async`` arguments. The two are intentionally orthogonal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import GenericAlias


class RoleTag(str, Enum):
    """Taxonomy of input roles the wizard / waterfall can elicit."""

    SCALAR = "scalar"
    CHOICE = "choice"
    REGISTRY_REF = "registry_ref"
    FACTORY = "factory"
    OPAQUE = "opaque"


@dataclass(frozen=True)
class RoleDescriptor:
    """
    Describes one ``__init__`` input of a scenario.

    Args:
        name (str): Argument name on the scenario's ``__init__``. Must be a
            valid Python identifier.
        description (str): Human-readable description shown in wizard prompts
            and ``--help`` output.
        tag (RoleTag): The elicitation strategy class. See module docstring.
        param_type (type | GenericAlias | None): Optional declared type for
            coercion / validation. ``None`` skips type enforcement; callers are
            then responsible for type correctness.
        choices (tuple[Any, ...] | None): For :attr:`RoleTag.CHOICE` roles,
            the allowed alternatives. ``None`` for other tags.
        default (Any): Default value when the caller does not supply one. May
            be ``None`` even for required roles (the wizard distinguishes via
            ``required``).
        required (bool): When True, the wizard must elicit a value; when False,
            ``default`` is used if no value is supplied.
    """

    name: str
    description: str
    tag: RoleTag
    param_type: type | GenericAlias | None = None
    choices: tuple[Any, ...] | None = None
    default: Any = None
    required: bool = True

    def __post_init__(self) -> None:
        """
        Validate structural invariants and normalize ``choices`` to a tuple.

        Raises:
            ValueError: If ``name`` isn't a valid identifier, if a CHOICE-tagged
                role has no choices, if a non-CHOICE role declares choices, or
                if an OPAQUE role is optional without a default.
        """
        if not self.name.isidentifier():
            raise ValueError(f"RoleDescriptor.name must be a valid Python identifier, got {self.name!r}.")

        if self.choices is not None and not isinstance(self.choices, tuple):
            object.__setattr__(self, "choices", tuple(self.choices))

        if self.tag is RoleTag.CHOICE:
            if not self.choices:
                raise ValueError(f"RoleDescriptor '{self.name}' tagged CHOICE must declare non-empty choices.")
        elif self.choices is not None:
            raise ValueError(
                f"RoleDescriptor '{self.name}' tagged {self.tag.value} must not declare choices "
                "(choices are only valid for CHOICE-tagged roles)."
            )

        if self.tag is RoleTag.OPAQUE and self.required is False and self.default is None:
            raise ValueError(
                f"RoleDescriptor '{self.name}' tagged OPAQUE cannot be optional without a default — "
                "opaque roles cannot be elicited from the CLI, so a non-None default is required."
            )
