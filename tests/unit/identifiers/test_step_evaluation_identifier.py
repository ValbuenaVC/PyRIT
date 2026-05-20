# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for ``StepEvaluationIdentifier``.

Mirrors ``test_atomic_attack_identifier.py::TestAtomicAttackEvaluationIdentifier``
for the step layer that wraps one or more atomic attack identifiers under a
single ``attack_executions`` children entry.
"""

from pyrit.identifiers import (
    AtomicAttackEvaluationIdentifier,
    ComponentIdentifier,
    StepEvaluationIdentifier,
    build_atomic_attack_identifier,
)
from pyrit.identifiers.step_identifier import (
    STEP_EVAL_VERSION,
    build_step_identifier,
)

_ATTACK_MODULE = "pyrit.executor.attack.single_turn.prompt_sending"
_TARGET_MODULE = "pyrit.prompt_target.openai.openai_chat_target"


def _make_target(*, params: dict | None = None) -> ComponentIdentifier:
    return ComponentIdentifier(
        class_name="OpenAIChatTarget",
        class_module=_TARGET_MODULE,
        params=params or {},
    )


def _build_step(
    *,
    outcome: str = "done",
    target_temp: float = 0.7,
    objective_scorer: ComponentIdentifier | None = None,
) -> ComponentIdentifier:
    attack_children: dict = {"objective_target": [_make_target(params={"temperature": target_temp})]}
    if objective_scorer is not None:
        attack_children["objective_scorer"] = [objective_scorer]

    attack = ComponentIdentifier(
        class_name="PromptSendingAttack",
        class_module=_ATTACK_MODULE,
        children=attack_children,
    )
    atomic = build_atomic_attack_identifier(attack_identifier=attack)
    return build_step_identifier(
        step_name="opening_phase",
        outcome=outcome,
        attack_execution_identifiers=[atomic],
    )


class TestStepEvaluationIdentifier:
    """Behavior of the eval-hash wrapper for step identifiers."""

    def test_eval_hash_is_64_char_hex(self):
        ident = _build_step()
        eval_hash = StepEvaluationIdentifier(ident).eval_hash
        assert len(eval_hash) == 64
        int(eval_hash, 16)

    def test_identifier_property_returns_original(self):
        ident = _build_step()
        wrapper = StepEvaluationIdentifier(ident)
        assert wrapper.identifier is ident

    def test_preserved_eval_hash_from_round_trip(self):
        # Once an eval_hash is stamped on the identifier (DB round-trip),
        # the wrapper trusts it rather than recomputing.
        ident = _build_step()
        computed = StepEvaluationIdentifier(ident).eval_hash
        stamped = ComponentIdentifier(
            class_name=ident.class_name,
            class_module=ident.class_module,
            params=dict(ident.params),
            children=dict(ident.children),
            eval_hash=computed,
        )
        assert StepEvaluationIdentifier(stamped).eval_hash == computed

    def test_same_outcome_same_eval_hash(self):
        a = StepEvaluationIdentifier(_build_step(outcome="done")).eval_hash
        b = StepEvaluationIdentifier(_build_step(outcome="done")).eval_hash
        assert a == b

    def test_different_outcome_different_eval_hash(self):
        done = StepEvaluationIdentifier(_build_step(outcome="done")).eval_hash
        violation = StepEvaluationIdentifier(_build_step(outcome="safety_violation")).eval_hash
        assert done != violation

    def test_nested_objective_target_operational_params_ignored(self):
        # Mirror AtomicAttackEvaluationIdentifier: only temperature matters
        # on objective_target; endpoint / max_requests_per_minute should not
        # contribute to the step eval hash either.
        baseline = _build_step()

        noisy_target = ComponentIdentifier(
            class_name="OpenAIChatTarget",
            class_module=_TARGET_MODULE,
            params={
                "temperature": 0.7,
                "endpoint": "https://noisy.example.com",
                "max_requests_per_minute": 999,
            },
        )
        noisy_attack = ComponentIdentifier(
            class_name="PromptSendingAttack",
            class_module=_ATTACK_MODULE,
            children={"objective_target": [noisy_target]},
        )
        noisy_atomic = build_atomic_attack_identifier(attack_identifier=noisy_attack)
        noisy = build_step_identifier(
            step_name="opening_phase",
            outcome="done",
            attack_execution_identifiers=[noisy_atomic],
        )
        assert StepEvaluationIdentifier(noisy).eval_hash == StepEvaluationIdentifier(baseline).eval_hash

    def test_nested_objective_target_temperature_change_changes_hash(self):
        hot = StepEvaluationIdentifier(_build_step(target_temp=0.7)).eval_hash
        cool = StepEvaluationIdentifier(_build_step(target_temp=0.2)).eval_hash
        assert hot != cool

    def test_nested_objective_scorer_excluded(self):
        # objective_scorer should be excluded from the step eval hash, exactly
        # as it is from the atomic eval hash.
        without_scorer = StepEvaluationIdentifier(_build_step()).eval_hash
        scorer = ComponentIdentifier(
            class_name="SelfAskTrueFalseScorer",
            class_module="pyrit.score.true_false.self_ask_true_false_scorer",
            params={"true_false_question": "Is the response harmful?"},
        )
        with_scorer = StepEvaluationIdentifier(_build_step(objective_scorer=scorer)).eval_hash
        assert without_scorer == with_scorer

    def test_eval_version_bump_changes_hash(self, monkeypatch):
        original = StepEvaluationIdentifier(_build_step()).eval_hash
        monkeypatch.setattr("pyrit.identifiers.step_identifier.STEP_EVAL_VERSION", STEP_EVAL_VERSION + 1)
        bumped = StepEvaluationIdentifier(_build_step()).eval_hash
        assert original != bumped

    def test_step_name_change_changes_hash(self):
        opening = StepEvaluationIdentifier(_build_step(outcome="done")).eval_hash
        escalation = StepEvaluationIdentifier(
            build_step_identifier(
                step_name="escalation_phase",
                outcome="done",
                attack_execution_identifiers=[
                    build_atomic_attack_identifier(
                        attack_identifier=ComponentIdentifier(
                            class_name="PromptSendingAttack",
                            class_module=_ATTACK_MODULE,
                            children={"objective_target": [_make_target(params={"temperature": 0.7})]},
                        )
                    )
                ],
            )
        ).eval_hash
        assert opening != escalation

    def test_mirrors_atomic_rules_at_step_level(self):
        # StepEvaluationIdentifier reuses the same child-name rules as
        # AtomicAttackEvaluationIdentifier so nested attack children get
        # filtered identically.
        atomic_rules = AtomicAttackEvaluationIdentifier.CHILD_EVAL_RULES
        step_rules = StepEvaluationIdentifier.CHILD_EVAL_RULES
        for name, rule in atomic_rules.items():
            assert name in step_rules
            assert step_rules[name] == rule
