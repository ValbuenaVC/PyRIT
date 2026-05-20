# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Phase 8a — coverage for ``RoleDescriptor`` + ``RoleTag``."""

from dataclasses import FrozenInstanceError

import pytest

from pyrit.scenario.core.input_schema import RoleDescriptor, RoleTag


class TestRoleTag:
    def test_role_tag_is_str_enum(self):
        """``RoleTag`` is a ``(str, Enum)`` so values are JSON-serializable."""
        assert isinstance(RoleTag.SCALAR.value, str)
        assert RoleTag.SCALAR == "scalar"

    def test_role_tag_members_complete(self):
        """The five canonical tags are present and distinct."""
        names = {tag.name for tag in RoleTag}
        assert names == {"SCALAR", "CHOICE", "REGISTRY_REF", "FACTORY", "OPAQUE"}


class TestRoleDescriptorConstruction:
    def test_minimal_scalar_role(self):
        role = RoleDescriptor(name="weakness_label", description="Label", tag=RoleTag.SCALAR)
        assert role.name == "weakness_label"
        assert role.tag is RoleTag.SCALAR
        assert role.required is True
        assert role.default is None
        assert role.choices is None

    def test_frozen_instance(self):
        role = RoleDescriptor(name="x", description="d", tag=RoleTag.SCALAR)
        with pytest.raises(FrozenInstanceError):
            role.name = "y"  # type: ignore[misc]

    def test_choice_role_with_choices(self):
        role = RoleDescriptor(
            name="mode",
            description="Operating mode",
            tag=RoleTag.CHOICE,
            choices=("fast", "slow"),
            param_type=str,
        )
        assert role.choices == ("fast", "slow")

    def test_choices_normalized_to_tuple(self):
        role = RoleDescriptor(
            name="mode",
            description="d",
            tag=RoleTag.CHOICE,
            choices=["a", "b"],  # type: ignore[arg-type]
        )
        assert role.choices == ("a", "b")
        assert isinstance(role.choices, tuple)


class TestRoleDescriptorValidation:
    def test_name_must_be_identifier(self):
        with pytest.raises(ValueError, match="valid Python identifier"):
            RoleDescriptor(name="not-a-name", description="d", tag=RoleTag.SCALAR)

    def test_name_with_space_rejected(self):
        with pytest.raises(ValueError, match="valid Python identifier"):
            RoleDescriptor(name="bad name", description="d", tag=RoleTag.SCALAR)

    def test_choice_without_choices_rejected(self):
        with pytest.raises(ValueError, match="must declare non-empty choices"):
            RoleDescriptor(name="mode", description="d", tag=RoleTag.CHOICE)

    def test_choice_with_empty_choices_rejected(self):
        with pytest.raises(ValueError, match="must declare non-empty choices"):
            RoleDescriptor(name="mode", description="d", tag=RoleTag.CHOICE, choices=())

    def test_non_choice_with_choices_rejected(self):
        with pytest.raises(ValueError, match="must not declare choices"):
            RoleDescriptor(name="x", description="d", tag=RoleTag.SCALAR, choices=("a",))

    def test_opaque_optional_without_default_rejected(self):
        with pytest.raises(ValueError, match="opaque roles cannot be elicited"):
            RoleDescriptor(name="sweep", description="d", tag=RoleTag.OPAQUE, required=False)

    def test_opaque_optional_with_default_allowed(self):
        sentinel = object()
        role = RoleDescriptor(
            name="sweep",
            description="d",
            tag=RoleTag.OPAQUE,
            required=False,
            default=sentinel,
        )
        assert role.default is sentinel

    def test_opaque_required_allowed(self):
        role = RoleDescriptor(name="sweep", description="d", tag=RoleTag.OPAQUE, required=True)
        assert role.required is True
        assert role.default is None
