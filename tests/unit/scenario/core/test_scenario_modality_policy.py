# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for ``Scenario.MODALITY_POLICY`` — plan-time enforcement of modality compatibility.

Validation runs inside ``initialize_async``, immediately after ``_build_atomic_attacks_async``
returns and before any attack is queued or any prompt is sent. The policy decides what happens
to an attack whose payload provably cannot reach its target or scorer.
"""

from __future__ import annotations

import logging
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from unit.mocks import get_mock_target
from unit.modality_profiles import TEXT_ONLY_MODALITIES, VISION_INPUT_MODALITIES

from pyrit.converter import QRCodeConverter
from pyrit.executor.attack import AttackConverterConfig, PromptSendingAttack
from pyrit.models import AttackSeedGroup, ComponentIdentifier, SeedObjective
from pyrit.prompt_normalizer import ConverterConfiguration
from pyrit.prompt_target.common.target_requirements import TargetRequirements
from pyrit.scenario.core import AtomicAttack, BaselineAttackPolicy, Scenario, ScenarioTechnique
from pyrit.scenario.core.attack_technique import AttackTechnique
from pyrit.scenario.core.dataset_configuration import DatasetConfiguration
from pyrit.scenario.core.modality_validation import ModalityPolicy, ModalityValidationError
from pyrit.score import Scorer

_TEST_SCORER_ID = ComponentIdentifier(class_name="MockScorer", class_module="tests.unit.scenario")


class _PolicyScenario(Scenario):
    """Minimal scenario returning a fixed list of atomic attacks."""

    BASELINE_ATTACK_POLICY: ClassVar[BaselineAttackPolicy] = BaselineAttackPolicy.Forbidden

    def __init__(self, *, atomic_attacks_to_return=None, **kwargs):
        class TestTechnique(ScenarioTechnique):
            TEST = ("test", {"concrete"}, "Test technique description.")
            ALL = ("all", {"all"})

            @classmethod
            def get_aggregate_tags(cls) -> set[str]:
                return {"all"}

        kwargs.setdefault("technique_class", TestTechnique)
        kwargs.setdefault("default_dataset_config", DatasetConfiguration())
        kwargs.setdefault("version", 1)
        if "objective_scorer" not in kwargs:
            scorer = MagicMock(spec=Scorer)
            scorer.get_identifier.return_value = _TEST_SCORER_ID
            scorer.get_scorer_metrics.return_value = None
            kwargs["objective_scorer"] = scorer
        super().__init__(**kwargs)
        self._atomic_attacks_to_return = atomic_attacks_to_return or []

    async def _resolve_seed_groups_by_dataset_async(self, *, apply_sampling: bool = True):
        return {}

    async def _build_atomic_attacks_async(self, *, context):
        return self._atomic_attacks_to_return


def _atomic(*, target, converters=None, name="atomic") -> AtomicAttack:
    """A real AtomicAttack sending one objective through an optional converter chain."""
    kwargs = {"objective_target": target}
    if converters is not None:
        kwargs["attack_converter_config"] = AttackConverterConfig(
            request_converters=ConverterConfiguration.from_converters(converters=list(converters))
        )
    return AtomicAttack(
        atomic_attack_name=name,
        attack_technique=AttackTechnique(attack=PromptSendingAttack(**kwargs)),
        seed_groups=[AttackSeedGroup(seeds=[SeedObjective(value=f"objective for {name}")])],
    )


def _incompatible(*, name="incompatible") -> AtomicAttack:
    """An attack whose converter emits an image into a text-only target."""
    target = get_mock_target(input_modalities=TEXT_ONLY_MODALITIES, output_modalities=TEXT_ONLY_MODALITIES)
    return _atomic(target=target, converters=[QRCodeConverter()], name=name)


def _compatible(*, name="compatible") -> AtomicAttack:
    """An attack whose converter emits an image into a target that accepts images."""
    target = get_mock_target(input_modalities=VISION_INPUT_MODALITIES, output_modalities=TEXT_ONLY_MODALITIES)
    return _atomic(target=target, converters=[QRCodeConverter()], name=name)


async def _initialize(scenario: Scenario, *, target=None) -> None:
    """Fill the parameter bag and initialize."""
    scenario.set_params_from_args(args={"objective_target": target or get_mock_target()})
    await scenario.initialize_async()


# ---------------------------------------------------------------------------
# Policy declaration
# ---------------------------------------------------------------------------
def test_modality_policy_defaults_to_skip():
    """Dropping unrunnable attacks is the default, matching BASELINE_ATTACK_POLICY's shape."""
    assert Scenario.MODALITY_POLICY is ModalityPolicy.SKIP


def test_modality_policy_is_overridable_per_scenario_class():
    """A scenario subclass can choose a stricter policy."""

    class _Strict(_PolicyScenario):
        MODALITY_POLICY: ClassVar[ModalityPolicy] = ModalityPolicy.RAISE

    assert _Strict.MODALITY_POLICY is ModalityPolicy.RAISE
    assert _PolicyScenario.MODALITY_POLICY is ModalityPolicy.SKIP


# ---------------------------------------------------------------------------
# SKIP
# ---------------------------------------------------------------------------
async def test_skip_drops_incompatible_and_keeps_compatible(patch_central_database):
    """The incompatible attack is removed; the compatible one survives."""
    scenario = _PolicyScenario(atomic_attacks_to_return=[_incompatible(), _compatible()])
    await _initialize(scenario)
    assert [attack.atomic_attack_name for attack in scenario._atomic_attacks] == ["compatible"]


async def test_skip_keeps_attacks_whose_compatibility_is_unknown(patch_central_database):
    """An attack against a target with unreadable capabilities is kept, never dropped."""
    scenario = _PolicyScenario(atomic_attacks_to_return=[_atomic(target=get_mock_target(), name="unknown")])
    await _initialize(scenario)
    assert [attack.atomic_attack_name for attack in scenario._atomic_attacks] == ["unknown"]


async def test_skip_excludes_dropped_attack_from_display_group_map(patch_central_database):
    """Filtering happens before the display-group map is built from the surviving attacks."""
    scenario = _PolicyScenario(atomic_attacks_to_return=[_incompatible(), _compatible()])
    await _initialize(scenario)
    assert "incompatible" not in scenario._display_group_map
    assert "compatible" in scenario._display_group_map


async def test_skip_excludes_dropped_attack_from_persisted_run_plan(patch_central_database):
    """The persisted plan records only the attacks that will actually run."""
    scenario = _PolicyScenario(atomic_attacks_to_return=[_incompatible(), _compatible()])
    await _initialize(scenario)
    [stored] = scenario._memory.get_scenario_results(scenario_result_ids=[scenario._scenario_result_id])
    planned = {group["atomic_attack_name"] for group in stored.metadata["run_plan"]["atomic_groups"]}
    assert planned == {"compatible"}


async def test_skip_logs_a_warning_naming_the_attack_and_reason(patch_central_database, caplog):
    """Dropping a whole atomic attack is loud even though the policy allows it."""
    scenario = _PolicyScenario(atomic_attacks_to_return=[_incompatible(), _compatible()])
    with caplog.at_level(logging.WARNING, logger="pyrit.scenario.core.scenario"):
        await _initialize(scenario)
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert any("incompatible" in record.getMessage() for record in warnings)
    assert any("image_path" in record.getMessage() for record in warnings)


async def test_skip_raises_when_every_attack_is_dropped(patch_central_database):
    """A run that would proceed with zero attacks is a failure, not a success."""
    scenario = _PolicyScenario(atomic_attacks_to_return=[_incompatible(name="a"), _incompatible(name="b")])
    with pytest.raises(ModalityValidationError, match="all"):
        await _initialize(scenario)


# ---------------------------------------------------------------------------
# WARN
# ---------------------------------------------------------------------------
async def test_warn_keeps_the_attack_and_logs(patch_central_database, caplog):
    """WARN surfaces the problem but lets the run proceed."""

    class _Warn(_PolicyScenario):
        MODALITY_POLICY: ClassVar[ModalityPolicy] = ModalityPolicy.WARN

    scenario = _Warn(atomic_attacks_to_return=[_incompatible()])
    with caplog.at_level(logging.WARNING, logger="pyrit.scenario.core.scenario"):
        await _initialize(scenario)
    assert [attack.atomic_attack_name for attack in scenario._atomic_attacks] == ["incompatible"]
    assert any("incompatible" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# RAISE
# ---------------------------------------------------------------------------
async def test_raise_aborts_initialization(patch_central_database):
    """RAISE stops the run before anything is persisted or queued."""

    class _Raise(_PolicyScenario):
        MODALITY_POLICY: ClassVar[ModalityPolicy] = ModalityPolicy.RAISE

    scenario = _Raise(atomic_attacks_to_return=[_incompatible(), _compatible()])
    with pytest.raises(ModalityValidationError) as excinfo:
        await _initialize(scenario)
    assert "incompatible" in str(excinfo.value)
    assert "image_path" in str(excinfo.value)


async def test_raise_error_is_catchable_as_value_error(patch_central_database):
    """Existing ``except ValueError`` handlers around initialize_async keep working."""

    class _Raise(_PolicyScenario):
        MODALITY_POLICY: ClassVar[ModalityPolicy] = ModalityPolicy.RAISE

    scenario = _Raise(atomic_attacks_to_return=[_incompatible()])
    with pytest.raises(ValueError):
        await _initialize(scenario)


async def test_raise_happens_before_any_prompt_is_sent(patch_central_database):
    """Fail fast: the target is never contacted when validation rejects the plan."""

    class _Raise(_PolicyScenario):
        MODALITY_POLICY: ClassVar[ModalityPolicy] = ModalityPolicy.RAISE

    attack = _incompatible()
    target = attack.attack_technique.attack.get_objective_target()
    scenario = _Raise(atomic_attacks_to_return=[attack])
    with pytest.raises(ModalityValidationError):
        await _initialize(scenario)
    assert target.send_prompt_async.call_count == 0


# ---------------------------------------------------------------------------
# Ordering against the existing static check
# ---------------------------------------------------------------------------
async def test_target_requirements_failure_precedes_modality_validation(patch_central_database):
    """
    ``TARGET_REQUIREMENTS`` is resolved before atomic attacks are built.

    A scenario whose target fails the static capability check reports that, not a derived
    modality verdict — the static misconfiguration is the more fundamental problem.
    """

    class _NeedsVideo(_PolicyScenario):
        TARGET_REQUIREMENTS: ClassVar[TargetRequirements] = TargetRequirements(
            required_input_modalities=frozenset({frozenset({"video_path"})})
        )
        MODALITY_POLICY: ClassVar[ModalityPolicy] = ModalityPolicy.RAISE

    scenario = _NeedsVideo(atomic_attacks_to_return=[_incompatible()])
    text_target = get_mock_target(input_modalities=TEXT_ONLY_MODALITIES)
    with pytest.raises(ValueError, match="video_path") as excinfo:
        await _initialize(scenario, target=text_target)
    assert not isinstance(excinfo.value, ModalityValidationError)
