# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
DatasetAdapter — converts SeedAttackGroup lists to Inspect AI datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyrit.inspect_bridge._imports import require_inspect_ai
from pyrit.models.seeds.seed_attack_group import SeedAttackGroup

if TYPE_CHECKING:
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
                exactly one objective.

        Raises:
            ValueError: If any element is not a ``SeedAttackGroup``.

        """
        for group in seed_groups:
            if not isinstance(group, SeedAttackGroup):
                raise ValueError(
                    f"DatasetAdapter requires SeedAttackGroup instances (each with exactly one objective), "
                    f"but received {type(group).__name__}. Plain SeedGroup without an objective is not supported."
                )
        self._seed_groups = list(seed_groups)

    def to_inspect_dataset(self) -> Any:
        """
        Convert the seed groups to an Inspect AI ``Dataset``.

        Each ``SeedAttackGroup`` maps to one ``Sample``:

        - ``input``: the first prompt value from the group (or empty string).
        - ``target``: the objective value.
        - ``metadata``: includes ``prompt_group_id`` and all prompt values.

        Returns:
            inspect_ai.dataset.Dataset: The Inspect dataset with one ``Sample``
            per seed group.

        """
        require_inspect_ai()
        from inspect_ai.dataset import MemoryDataset, Sample

        samples: list[Sample] = []
        for group in self._seed_groups:
            objective_text = group.objective.value
            prompt_values = [p.value for p in group.prompts]
            input_text = prompt_values[0] if prompt_values else ""
            group_id = str(group.seeds[0].prompt_group_id) if group.seeds else None

            sample = Sample(
                input=input_text,
                target=objective_text,
                metadata={
                    "prompt_group_id": group_id,
                    "all_prompts": prompt_values,
                    "objective": objective_text,
                },
            )
            samples.append(sample)

        return MemoryDataset(samples=samples)

    @classmethod
    def from_seed_dataset(cls, *, seed_dataset: SeedDataset) -> DatasetAdapter:
        """
        Build a ``DatasetAdapter`` from a ``SeedDataset``.

        Extracts ``SeedAttackGroup`` instances from the dataset's seed groups.
        Raises if no attack groups are found.

        Args:
            seed_dataset (SeedDataset): The seed dataset to convert.

        Returns:
            DatasetAdapter: A new adapter wrapping the validated attack groups.

        Raises:
            ValueError: If the dataset contains no ``SeedAttackGroup`` instances.

        """
        attack_groups = [g for g in seed_dataset.seed_groups if isinstance(g, SeedAttackGroup)]
        if not attack_groups:
            raise ValueError(
                "SeedDataset contains no SeedAttackGroup instances (groups with exactly one objective). "
                "Ensure the dataset has seeds with a SeedObjective."
            )
        return cls(seed_groups=attack_groups)
