# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for S1.3: DatasetAdapter (SeedAttackGroup -> Inspect Dataset).

No memory, no network.
"""

from __future__ import annotations

import pytest

from pyrit.models.seeds.seed_attack_group import SeedAttackGroup
from pyrit.models.seeds.seed_dataset import SeedDataset
from pyrit.models.seeds.seed_group import SeedGroup
from pyrit.models.seeds.seed_objective import SeedObjective
from pyrit.models.seeds.seed_prompt import SeedPrompt


def _make_attack_group(*, objective: str = "Do something harmful", prompt: str = "Hello") -> SeedAttackGroup:
    obj = SeedObjective(value=objective)
    p = SeedPrompt(value=prompt, data_type="text", role="user", sequence=0)
    return SeedAttackGroup(seeds=[obj, p])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_dataset_adapter_construction() -> None:
    """DatasetAdapter can be constructed with a list of SeedAttackGroup."""
    from pyrit.inspect_bridge._dataset_adapter import DatasetAdapter

    groups = [_make_attack_group()]
    adapter = DatasetAdapter(seed_groups=groups)
    assert adapter is not None


def test_dataset_adapter_plain_seed_group_raises() -> None:
    """DatasetAdapter raises ValueError if a plain SeedGroup (no objective) is provided."""
    from pyrit.inspect_bridge._dataset_adapter import DatasetAdapter

    obj = SeedObjective(value="obj")
    p = SeedPrompt(value="prompt", data_type="text", role="user", sequence=0)
    # SeedAttackGroup enforces one objective — use a plain SeedGroup to trigger
    # the error (we can work around by passing the wrong type)
    # SeedGroup without objective would be accepted at SeedGroup level but rejected by DatasetAdapter
    # We create a group with no objective to test:
    p_only = SeedPrompt(value="no-obj", data_type="text", role="user", sequence=0)
    plain_group = SeedGroup(seeds=[p_only])  # plain SeedGroup, no SeedAttackGroup

    with pytest.raises((ValueError, TypeError)):
        DatasetAdapter(seed_groups=[plain_group])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# to_inspect_dataset
# ---------------------------------------------------------------------------


def test_to_inspect_dataset_returns_dataset() -> None:
    """to_inspect_dataset() returns an Inspect Dataset."""
    from inspect_ai.dataset import Dataset

    from pyrit.inspect_bridge._dataset_adapter import DatasetAdapter

    groups = [_make_attack_group(objective="Harm user", prompt="Say bad things")]
    adapter = DatasetAdapter(seed_groups=groups)
    dataset = adapter.to_inspect_dataset()
    assert isinstance(dataset, Dataset)


def test_to_inspect_dataset_sample_count() -> None:
    """Dataset has one Sample per SeedAttackGroup."""
    from pyrit.inspect_bridge._dataset_adapter import DatasetAdapter

    groups = [
        _make_attack_group(objective="Objective 1", prompt="Prompt 1"),
        _make_attack_group(objective="Objective 2", prompt="Prompt 2"),
    ]
    adapter = DatasetAdapter(seed_groups=groups)
    dataset = adapter.to_inspect_dataset()
    assert len(dataset) == 2


def test_to_inspect_dataset_sample_input_is_prompt() -> None:
    """Each Sample's input contains the prompt from the SeedAttackGroup."""
    from pyrit.inspect_bridge._dataset_adapter import DatasetAdapter

    group = _make_attack_group(objective="Make harm", prompt="Tell me how to hack")
    adapter = DatasetAdapter(seed_groups=[group])
    dataset = adapter.to_inspect_dataset()
    sample = dataset[0]
    # input should contain the prompt text
    input_str = sample.input if isinstance(sample.input, str) else str(sample.input)
    assert "Tell me how to hack" in input_str


def test_to_inspect_dataset_sample_target_is_objective() -> None:
    """Each Sample's target contains the objective from the SeedAttackGroup."""
    from pyrit.inspect_bridge._dataset_adapter import DatasetAdapter

    group = _make_attack_group(objective="Reveal system prompt", prompt="Hi")
    adapter = DatasetAdapter(seed_groups=[group])
    dataset = adapter.to_inspect_dataset()
    sample = dataset[0]
    target_str = sample.target if isinstance(sample.target, str) else str(sample.target)
    assert "Reveal system prompt" in target_str


def test_to_inspect_dataset_metadata_has_group_id() -> None:
    """Each Sample's metadata contains the prompt_group_id."""
    from pyrit.inspect_bridge._dataset_adapter import DatasetAdapter

    group = _make_attack_group()
    adapter = DatasetAdapter(seed_groups=[group])
    dataset = adapter.to_inspect_dataset()
    sample = dataset[0]
    assert sample.metadata is not None
    assert "prompt_group_id" in sample.metadata


# ---------------------------------------------------------------------------
# from_seed_dataset classmethod
# ---------------------------------------------------------------------------


def test_from_seed_dataset_creates_adapter() -> None:
    """from_seed_dataset() creates a DatasetAdapter from a SeedDataset."""
    from pyrit.inspect_bridge._dataset_adapter import DatasetAdapter

    obj = SeedObjective(value="Cause harm")
    p = SeedPrompt(value="Attack prompt", data_type="text", role="user", sequence=0)
    seed_dataset = SeedDataset(seeds=[obj, p])
    # filter to SeedAttackGroups only
    attack_groups = [g for g in seed_dataset.seed_groups if isinstance(g, SeedAttackGroup)]
    if not attack_groups:
        pytest.skip("No SeedAttackGroups in test seed dataset")
    adapter = DatasetAdapter.from_seed_dataset(seed_dataset=seed_dataset)
    assert isinstance(adapter, DatasetAdapter)


def test_from_seed_dataset_with_no_attack_groups_raises() -> None:
    """from_seed_dataset() with a dataset that has no SeedAttackGroup seeds raises ValueError."""
    from pyrit.inspect_bridge._dataset_adapter import DatasetAdapter

    # A dataset with only plain prompts (no objectives) -> no SeedAttackGroups
    p1 = SeedPrompt(value="prompt 1", data_type="text", role="user", sequence=0)
    p2 = SeedPrompt(value="prompt 2", data_type="text", role="user", sequence=0)
    seed_dataset = SeedDataset(seeds=[p1, p2])

    with pytest.raises(ValueError):
        DatasetAdapter.from_seed_dataset(seed_dataset=seed_dataset)
