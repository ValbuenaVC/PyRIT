# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for :mod:`pyrit.scenario.composite.scenario_pipeline`.

Pins both the static surface (dataclass frozenness, input schema shape,
construction validation) and the dynamic behavior (phase dispatch order,
conditional skipping, predicate context, target/labels inheritance) of the
``ScenarioPipeline`` composition primitive.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.identifiers import ComponentIdentifier
from pyrit.scenario import (
    ConditionalPhaseSpec,
    PhaseExecution,
    PhaseSpec,
    PipelineContext,
    Scenario,
    ScenarioPipeline,
    ScenarioPipelineStrategy,
)
from pyrit.scenario.composite.scenario_pipeline import (
    _PHASE_OUTCOME_COMPLETED,
    _PHASE_OUTCOME_SKIPPED,
    _PipelineNoopScorer,
    _ScenarioPipelinePhaseStep,
)
from pyrit.scenario.core.input_schema import RoleTag
from pyrit.score import Scorer

_PIPELINE_SCORER_ID = ComponentIdentifier(
    class_name="MockScorer",
    class_module="tests.unit.scenario.composite",
)


def _make_inner_scenario(name: str = "inner") -> MagicMock:
    """Build a ``MagicMock(spec=Scenario)`` that satisfies the pipeline contract."""
    inner = MagicMock(spec=Scenario)
    inner.name = name
    fake_result = MagicMock()
    fake_result.id = f"result-{name}"
    inner.initialize_async = AsyncMock()
    inner.run_async = AsyncMock(return_value=fake_result)
    return inner


def _phase_spec(
    name: str,
    *,
    inner: MagicMock | None = None,
    init_kwargs: dict[str, Any] | None = None,
) -> tuple[PhaseSpec, MagicMock]:
    """Build a ``PhaseSpec`` whose factory returns the supplied (or fresh) inner mock."""
    inner = inner if inner is not None else _make_inner_scenario(name)
    spec = PhaseSpec(
        name=name,
        scenario_factory=lambda: inner,
        init_async_kwargs=init_kwargs or {},
    )
    return spec, inner


def _conditional_phase_spec(
    name: str,
    *,
    skip_when,
    inner: MagicMock | None = None,
    init_kwargs: dict[str, Any] | None = None,
) -> tuple[ConditionalPhaseSpec, MagicMock]:
    inner = inner if inner is not None else _make_inner_scenario(name)
    spec = ConditionalPhaseSpec(
        name=name,
        scenario_factory=lambda: inner,
        init_async_kwargs=init_kwargs or {},
        skip_when=skip_when,
    )
    return spec, inner


@pytest.fixture
def mock_objective_target():
    target = MagicMock()
    target.get_identifier.return_value = ComponentIdentifier(
        class_name="MockTarget",
        class_module="tests.unit.scenario.composite",
    )
    return target


@pytest.fixture
def mock_scorer():
    scorer = MagicMock(spec=Scorer)
    scorer.get_identifier.return_value = _PIPELINE_SCORER_ID
    scorer.get_scorer_metrics.return_value = None
    return scorer


# --------------------------------------------------------------------------- #
# Static surface
# --------------------------------------------------------------------------- #


class TestPhaseSpecDataclasses:
    """Dataclass shape: frozen, default factory, conditional inheritance."""

    def test_phase_spec_is_frozen(self):
        spec, _ = _phase_spec("p")
        with pytest.raises((AttributeError, Exception)):
            spec.name = "renamed"  # type: ignore[misc]

    def test_phase_spec_init_async_kwargs_defaults_to_empty(self):
        inner = _make_inner_scenario()
        spec = PhaseSpec(name="p", scenario_factory=lambda: inner)
        assert dict(spec.init_async_kwargs) == {}

    def test_conditional_phase_spec_inherits_phase_spec_fields(self):
        spec, _ = _conditional_phase_spec("p", skip_when=lambda _ctx: True)
        assert isinstance(spec, PhaseSpec)
        assert spec.name == "p"

    def test_conditional_phase_spec_default_skip_when_runs_phase(self):
        inner = _make_inner_scenario()
        spec = ConditionalPhaseSpec(name="p", scenario_factory=lambda: inner)
        # Default skip_when returns False (= run) for any context.
        assert spec.skip_when(PipelineContext()) is False

    def test_phase_execution_is_frozen(self):
        execution = PhaseExecution(name="p", outcome=_PHASE_OUTCOME_COMPLETED, scenario_result=None)
        with pytest.raises((AttributeError, Exception)):
            execution.name = "renamed"  # type: ignore[misc]

    def test_pipeline_context_is_frozen_with_tuple_defaults(self):
        context = PipelineContext()
        assert context.completed_phase_names == ()
        assert context.completed_phase_outcomes == ()
        assert context.phase_executions == ()
        with pytest.raises((AttributeError, Exception)):
            context.completed_phase_names = ("x",)  # type: ignore[misc]


class TestScenarioPipelineStrategy:
    """The single-member enum that satisfies the base ``Scenario`` contract."""

    def test_default_member_exists(self):
        assert ScenarioPipelineStrategy.DEFAULT.value == "default"

    def test_aggregate_tags_contains_all(self):
        assert "all" in ScenarioPipelineStrategy.get_aggregate_tags()


# --------------------------------------------------------------------------- #
# Construction validation
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("patch_central_database")
class TestConstruction:
    """Constructor input validation."""

    def test_rejects_empty_phases(self):
        with pytest.raises(ValueError, match="at least one PhaseSpec"):
            ScenarioPipeline(phases=[])

    def test_rejects_duplicate_names(self):
        spec_a, _ = _phase_spec("a")
        spec_b, _ = _phase_spec("a")
        with pytest.raises(ValueError, match="must be unique"):
            ScenarioPipeline(phases=[spec_a, spec_b])

    def test_rejects_scenario_result_id(self):
        spec, _ = _phase_spec("a")
        with pytest.raises(ValueError, match="does not support pipeline-level resume"):
            ScenarioPipeline(phases=[spec], scenario_result_id="some-id")

    def test_constructs_with_single_phase_and_default_scorer(self):
        spec, _ = _phase_spec("only")
        pipeline = ScenarioPipeline(phases=[spec])
        assert pipeline.phases == (spec,)
        # Default scorer is the deterministic no-op stand-in.
        assert isinstance(pipeline._objective_scorer, _PipelineNoopScorer)

    def test_constructs_with_explicit_scorer(self, mock_scorer):
        spec, _ = _phase_spec("only")
        pipeline = ScenarioPipeline(phases=[spec], objective_scorer=mock_scorer)
        assert pipeline._objective_scorer is mock_scorer

    def test_name_defaults_to_class_name(self):
        spec, _ = _phase_spec("only")
        assert ScenarioPipeline(phases=[spec]).name == "ScenarioPipeline"

    def test_name_override_wins(self):
        spec, _ = _phase_spec("only")
        assert ScenarioPipeline(phases=[spec], name="My Pipeline").name == "My Pipeline"


# --------------------------------------------------------------------------- #
# Input schema
# --------------------------------------------------------------------------- #


class TestInputSchema:
    """``input_schema()`` exposes the two pipeline-level roles."""

    def test_schema_lists_phases_and_name(self):
        schema = ScenarioPipeline.input_schema()
        names = [r.name for r in schema]
        assert names == ["phases", "name"]

    def test_phases_role_is_opaque_and_required(self):
        phases_role = next(r for r in ScenarioPipeline.input_schema() if r.name == "phases")
        assert phases_role.tag is RoleTag.OPAQUE
        assert phases_role.required is True

    def test_name_role_is_scalar_and_optional(self):
        name_role = next(r for r in ScenarioPipeline.input_schema() if r.name == "name")
        assert name_role.tag is RoleTag.SCALAR
        assert name_role.required is False


# --------------------------------------------------------------------------- #
# Phase step (custom ScenarioStep)
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("patch_central_database")
class TestPhaseStep:
    """The custom ``_ScenarioPipelinePhaseStep`` orchestrator-facing surface."""

    def test_duck_typed_atomic_attack_attributes(self):
        spec, _ = _phase_spec("dummy")
        pipeline = ScenarioPipeline(phases=[spec])
        step = _ScenarioPipelinePhaseStep(spec=spec, index=0, pipeline=pipeline)
        assert step.atomic_attack_name == "dummy"
        assert step.display_group == "dummy"
        assert step.seed_groups == []
        assert step.objectives == []

    def test_technique_eval_hash_sentinel_is_phase_scoped(self):
        spec, _ = _phase_spec("alpha")
        pipeline = ScenarioPipeline(phases=[spec])
        step = _ScenarioPipelinePhaseStep(spec=spec, index=0, pipeline=pipeline)
        # The sentinel namespaces by phase name so it can never collide with a real eval hash.
        assert step.technique_eval_hash == "pipeline-phase::alpha"

    def test_graph_artifact_opaque_marker_present(self):
        # Pipeline phases close over a callable factory; opacity is the correct
        # contract for Phase 8g GraphArtifact serialization.
        assert _ScenarioPipelinePhaseStep.GRAPH_ARTIFACT_OPAQUE is True

    def test_identifier_is_deterministic(self):
        spec, _ = _phase_spec("alpha")
        pipeline = ScenarioPipeline(phases=[spec])
        s1 = _ScenarioPipelinePhaseStep(spec=spec, index=0, pipeline=pipeline)
        s2 = _ScenarioPipelinePhaseStep(spec=spec, index=0, pipeline=pipeline)
        assert s1.get_identifier().hash == s2.get_identifier().hash

    def test_identifier_differs_by_index(self):
        spec, _ = _phase_spec("alpha")
        pipeline = ScenarioPipeline(phases=[spec])
        s_a = _ScenarioPipelinePhaseStep(spec=spec, index=0, pipeline=pipeline)
        s_b = _ScenarioPipelinePhaseStep(spec=spec, index=1, pipeline=pipeline)
        assert s_a.get_identifier().hash != s_b.get_identifier().hash

    def test_filter_seed_groups_is_noop(self):
        spec, _ = _phase_spec("alpha")
        pipeline = ScenarioPipeline(phases=[spec])
        step = _ScenarioPipelinePhaseStep(spec=spec, index=0, pipeline=pipeline)
        # No raise; pipeline phase steps own no seed groups at the pipeline level.
        step.filter_seed_groups_by_objectives(remaining_objectives=["o1"])
        step.drop_seed_groups_with_hashes(hashes={"h1"})


# --------------------------------------------------------------------------- #
# Execution graph build
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("patch_central_database")
class TestBuildExecutionGraph:
    """Linear int-state graph construction."""

    async def test_graph_has_one_state_per_phase(self, mock_objective_target):
        spec_a, _ = _phase_spec("a")
        spec_b, _ = _phase_spec("b")
        spec_c, _ = _phase_spec("c")
        pipeline = ScenarioPipeline(phases=[spec_a, spec_b, spec_c])
        await pipeline.initialize_async(objective_target=mock_objective_target)

        graph = pipeline._build_execution_graph()
        assert graph.policy.initial_state == 0
        assert graph.policy.terminal_states == frozenset({3})
        for state in range(3):
            assert callable(graph.policy.get_action(state=state))

    async def test_explicit_steps_override_atomic_attacks(self, mock_objective_target):
        spec_a, _ = _phase_spec("a")
        spec_b, _ = _phase_spec("b")
        pipeline = ScenarioPipeline(phases=[spec_a, spec_b])
        await pipeline.initialize_async(objective_target=mock_objective_target)

        single_step = _ScenarioPipelinePhaseStep(spec=spec_a, index=0, pipeline=pipeline)
        graph = pipeline._build_execution_graph(steps=[single_step])
        assert graph.policy.terminal_states == frozenset({1})

    async def test_build_raises_with_empty_steps(self, mock_objective_target):
        spec, _ = _phase_spec("a")
        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(objective_target=mock_objective_target)
        with pytest.raises(ValueError, match="cannot build an execution graph"):
            pipeline._build_execution_graph(steps=[])

    async def test_build_resets_phase_executions_log(self, mock_objective_target):
        spec, _ = _phase_spec("a")
        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(objective_target=mock_objective_target)

        pipeline._phase_executions.append(
            PhaseExecution(name="stale", outcome=_PHASE_OUTCOME_COMPLETED, scenario_result=None)
        )
        pipeline._build_execution_graph()
        assert pipeline._phase_executions == []


# --------------------------------------------------------------------------- #
# End-to-end execution
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("patch_central_database")
class TestPipelineExecution:
    """End-to-end ``run_async`` behavior with mocked inner scenarios."""

    async def test_single_phase_runs_and_records_execution(self, mock_objective_target):
        spec, inner = _phase_spec("only")
        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(objective_target=mock_objective_target)
        await pipeline.run_async()

        inner.initialize_async.assert_awaited_once()
        inner.run_async.assert_awaited_once()
        assert len(pipeline.phase_executions) == 1
        execution = pipeline.phase_executions[0]
        assert execution.name == "only"
        assert execution.outcome == _PHASE_OUTCOME_COMPLETED
        assert execution.scenario_result is inner.run_async.return_value

    async def test_two_phases_run_in_declaration_order(self, mock_objective_target):
        order: list[str] = []

        def _make_recording_inner(name: str):
            inner = _make_inner_scenario(name)

            async def _record_init(**_kwargs):
                order.append(f"init:{name}")

            async def _record_run():
                order.append(f"run:{name}")
                return inner.run_async.return_value

            inner.initialize_async.side_effect = _record_init
            inner.run_async.side_effect = _record_run
            return inner

        inner_a = _make_recording_inner("a")
        inner_b = _make_recording_inner("b")
        spec_a, _ = _phase_spec("a", inner=inner_a)
        spec_b, _ = _phase_spec("b", inner=inner_b)

        pipeline = ScenarioPipeline(phases=[spec_a, spec_b])
        await pipeline.initialize_async(objective_target=mock_objective_target)
        await pipeline.run_async()

        assert order == ["init:a", "run:a", "init:b", "run:b"]
        assert [e.name for e in pipeline.phase_executions] == ["a", "b"]
        assert all(e.outcome == _PHASE_OUTCOME_COMPLETED for e in pipeline.phase_executions)

    async def test_conditional_phase_skipped_does_not_call_factory(self, mock_objective_target):
        factory_calls: list[str] = []
        inner = _make_inner_scenario("skipme")

        def _factory():
            factory_calls.append("called")
            return inner

        spec = ConditionalPhaseSpec(
            name="skipme",
            scenario_factory=_factory,
            skip_when=lambda _ctx: True,
        )
        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(objective_target=mock_objective_target)
        await pipeline.run_async()

        assert factory_calls == []
        inner.initialize_async.assert_not_called()
        inner.run_async.assert_not_called()
        assert len(pipeline.phase_executions) == 1
        assert pipeline.phase_executions[0].outcome == _PHASE_OUTCOME_SKIPPED
        assert pipeline.phase_executions[0].scenario_result is None

    async def test_conditional_phase_runs_when_predicate_returns_false(self, mock_objective_target):
        spec, inner = _conditional_phase_spec("runme", skip_when=lambda _ctx: False)
        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(objective_target=mock_objective_target)
        await pipeline.run_async()

        inner.run_async.assert_awaited_once()
        assert pipeline.phase_executions[0].outcome == _PHASE_OUTCOME_COMPLETED

    async def test_predicate_receives_context_of_prior_phases(self, mock_objective_target):
        captured: list[PipelineContext] = []

        spec_a, _ = _phase_spec("a")
        spec_b, _ = _conditional_phase_spec(
            "b",
            skip_when=lambda ctx: (captured.append(ctx), False)[1],
        )
        spec_c, _ = _conditional_phase_spec(
            "c",
            skip_when=lambda ctx: (captured.append(ctx), True)[1],
        )

        pipeline = ScenarioPipeline(phases=[spec_a, spec_b, spec_c])
        await pipeline.initialize_async(objective_target=mock_objective_target)
        await pipeline.run_async()

        assert len(captured) == 2
        # When phase b evaluates, only phase a has executed.
        ctx_b = captured[0]
        assert ctx_b.completed_phase_names == ("a",)
        assert ctx_b.completed_phase_outcomes == (_PHASE_OUTCOME_COMPLETED,)
        # When phase c evaluates, phases a and b have both executed (b ran).
        ctx_c = captured[1]
        assert ctx_c.completed_phase_names == ("a", "b")
        assert ctx_c.completed_phase_outcomes == (_PHASE_OUTCOME_COMPLETED, _PHASE_OUTCOME_COMPLETED)

    async def test_pipeline_objective_target_flows_to_inner_scenario(self, mock_objective_target):
        spec, inner = _phase_spec("p")
        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(objective_target=mock_objective_target)
        await pipeline.run_async()

        inner.initialize_async.assert_awaited_once()
        call_kwargs = inner.initialize_async.await_args.kwargs
        assert call_kwargs["objective_target"] is mock_objective_target

    async def test_pipeline_memory_labels_flow_to_inner_scenario(self, mock_objective_target):
        spec, inner = _phase_spec("p")
        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(
            objective_target=mock_objective_target,
            memory_labels={"campaign": "alpha"},
        )
        await pipeline.run_async()

        call_kwargs = inner.initialize_async.await_args.kwargs
        assert call_kwargs["memory_labels"] == {"campaign": "alpha"}

    async def test_init_async_kwargs_override_pipeline_target(self, mock_objective_target):
        custom_target = MagicMock()
        custom_target.get_identifier.return_value = ComponentIdentifier(
            class_name="OverrideTarget",
            class_module="tests.unit.scenario.composite",
        )
        spec, inner = _phase_spec("p", init_kwargs={"objective_target": custom_target})

        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(objective_target=mock_objective_target)
        await pipeline.run_async()

        call_kwargs = inner.initialize_async.await_args.kwargs
        assert call_kwargs["objective_target"] is custom_target

    async def test_factory_returning_non_scenario_raises_type_error(self, mock_objective_target):
        spec = PhaseSpec(name="bad", scenario_factory=lambda: "not a scenario")
        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(objective_target=mock_objective_target)
        with pytest.raises(TypeError, match="'bad'"):
            await pipeline.run_async()
