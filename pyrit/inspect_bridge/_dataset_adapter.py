# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
DatasetAdapter — converts SeedAttackGroup lists to Inspect AI datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrit.models.seeds.seed_attack_group import SeedAttackGroup
    from pyrit.models.seeds.seed_dataset import SeedDataset


class DatasetAdapter:
    """
    Converts a list of ``SeedAttackGroup`` instances to an Inspect AI ``Dataset``.

    ``SeedAttackGroup`` is required because each instance guarantees exactly one
    ``SeedObjective``, which maps to one ``AtomicAttack``. A plain ``SeedGroup``
    without an objective is rejected with a ``ValueError``.
    """

    def __init__(self, *, seed_groups: list[SeedAttackGroup]) -> None:
        """
        Initialize the DatasetAdapter.

        Args:
            seed_groups (list[SeedAttackGroup]): Attack seed groups, each with
                exactly one objective. An empty list raises ``ValueError``.

        Raises:
            ValueError: If ``seed_groups`` is empty.

        """
        raise NotImplementedError

    def to_inspect_dataset(self) -> Any:
        """
        Convert the seed groups to an Inspect AI ``Dataset``.

        Returns:
            inspect_ai.dataset.Dataset: The Inspect dataset with one ``Sample``
            per seed group.

        """
        raise NotImplementedError

    @classmethod
    def from_seed_dataset(cls, *, seed_dataset: SeedDataset) -> DatasetAdapter:
        """
        Build a ``DatasetAdapter`` from a ``SeedDataset``.

        Validates that every group in the dataset is a ``SeedAttackGroup`` with
        exactly one objective; raises early if any group lacks one.

        Args:
            seed_dataset (SeedDataset): The seed dataset to convert.

        Returns:
            DatasetAdapter: A new adapter wrapping the validated attack groups.

        Raises:
            ValueError: If any group in the dataset is not a ``SeedAttackGroup``
                or lacks an objective.

        """
        raise NotImplementedError
