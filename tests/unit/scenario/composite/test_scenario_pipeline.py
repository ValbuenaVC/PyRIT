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

    def test_set_scenario_result_id_is_noop_stub(self):
        """Forward-compatibility with R1 (drop-isinstance refactor).

        Today the base ``Scenario._execute_scenario_async`` guards
        ``set_scenario_result_id`` with ``isinstance(_step, AtomicAttack)``,
        so this method is unreachable from the orchestrator. R1 plans to
        collapse that branch and dispatch uniformly via ``process_async``;
        when it does, every non-AtomicAttack ``ScenarioStep`` must expose
        ``set_scenario_result_id``. This regression pins the stub so that
        contract change doesn't silently introduce an ``AttributeError``
        for pipeline phase steps.
        """
        spec, _ = _phase_spec("alpha")
        pipeline = ScenarioPipeline(phases=[spec])
        step = _ScenarioPipelinePhaseStep(spec=spec, index=0, pipeline=pipeline)
        # Stub should accept either a real id or None without raising.
        step.set_scenario_result_id("any-id")
        step.set_scenario_result_id(None)


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

    async def test_factory_returning_duck_typed_non_scenario_class_raises_type_error(self, mock_objective_target):
        """Pin that the guard uses :func:`isinstance`, not duck-typing.

        A class exposing ``initialize_async``/``run_async`` methods but not
        inheriting from :class:`Scenario` must be rejected. This guards against
        a future refactor that swaps ``isinstance(inner_scenario, Scenario)``
        for an attribute-presence check, which would silently accept duck-typed
        objects whose ``run_async`` returns an incompatible result shape.
        """

        class _DuckScenario:
            async def initialize_async(self, **kwargs: Any) -> None:
                pass

            async def run_async(self, **kwargs: Any) -> Any:
                return None

        spec = PhaseSpec(name="duck", scenario_factory=lambda: _DuckScenario())
        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(objective_target=mock_objective_target)
        with pytest.raises(TypeError, match="'duck'"):
            await pipeline.run_async()

    async def test_factory_returning_none_raises_type_error(self, mock_objective_target):
        """A factory that returns ``None`` is a common typo (missing ``return``)."""
        spec = PhaseSpec(name="forgot_return", scenario_factory=lambda: None)
        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(objective_target=mock_objective_target)
        with pytest.raises(TypeError, match="'forgot_return'"):
            await pipeline.run_async()


# --------------------------------------------------------------------------- #
# Failure modes — exception propagation from inner lifecycle and predicates
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("patch_central_database")
class TestPipelineFailureModes:
    """Pin how exceptions raised by inner-scenario lifecycle and predicates surface.

    Covers the gaps Duck #1 (gpt-5.3-codex) flagged in the post-R5 review:
    inner ``initialize_async`` failures, inner ``run_async`` failures, and
    predicate failures. All three should propagate as exceptions out of
    ``pipeline.run_async()`` after marking the scenario FAILED in memory, not
    be silently caught.
    """

    async def test_inner_initialize_async_exception_propagates(self, mock_objective_target):
        inner = _make_inner_scenario("blowup")
        inner.initialize_async.side_effect = RuntimeError("init exploded")
        spec = PhaseSpec(name="blowup", scenario_factory=lambda: inner)

        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(objective_target=mock_objective_target)

        with pytest.raises(RuntimeError, match="init exploded"):
            await pipeline.run_async()
        # run_async should never have been reached.
        inner.run_async.assert_not_called()

    async def test_inner_run_async_exception_propagates(self, mock_objective_target):
        inner = _make_inner_scenario("midflight")
        inner.run_async.side_effect = RuntimeError("run exploded")
        spec = PhaseSpec(name="midflight", scenario_factory=lambda: inner)

        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(objective_target=mock_objective_target)

        with pytest.raises(RuntimeError, match="run exploded"):
            await pipeline.run_async()
        # initialize_async ran, run_async raised.
        inner.initialize_async.assert_awaited_once()

    async def test_predicate_exception_propagates(self, mock_objective_target):
        inner = _make_inner_scenario("downstream")

        spec_a, inner_a = _phase_spec("a")
        spec_b = ConditionalPhaseSpec(
            name="b_broken_predicate",
            scenario_factory=lambda: inner,
            skip_when=lambda _ctx: (_ for _ in ()).throw(ValueError("predicate logic bug")),
        )

        pipeline = ScenarioPipeline(phases=[spec_a, spec_b])
        await pipeline.initialize_async(objective_target=mock_objective_target)

        with pytest.raises(ValueError, match="predicate logic bug"):
            await pipeline.run_async()
        # Phase a completed before the broken predicate was evaluated.
        inner_a.run_async.assert_awaited_once()
        # The broken-predicate phase's factory was never called.
        inner.run_async.assert_not_called()

    async def test_failure_in_later_phase_records_earlier_phases_executions(self, mock_objective_target):
        """When phase N fails, phase_executions for phases 1..N-1 should still be recorded."""
        spec_a, inner_a = _phase_spec("alpha")
        inner_b = _make_inner_scenario("beta")
        inner_b.run_async.side_effect = RuntimeError("phase 2 down")
        spec_b = PhaseSpec(name="beta", scenario_factory=lambda: inner_b)

        pipeline = ScenarioPipeline(phases=[spec_a, spec_b])
        await pipeline.initialize_async(objective_target=mock_objective_target)

        with pytest.raises(RuntimeError, match="phase 2 down"):
            await pipeline.run_async()

        # The in-memory log captures the completed phase before the crash.
        assert len(pipeline.phase_executions) == 1
        assert pipeline.phase_executions[0].name == "alpha"
        assert pipeline.phase_executions[0].outcome == _PHASE_OUTCOME_COMPLETED


# --------------------------------------------------------------------------- #
# Artifact round-trip limitation (v1 contract pin)
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("patch_central_database")
class TestArtifactRoundTripNotSupported:
    """Pin the v1 contract that pipelines cannot round-trip via graph artifacts.

    A :class:`PhaseSpec` holds a ``Callable`` scenario factory and a
    :class:`types.MappingProxyType` ``init_async_kwargs`` (when defaulted) —
    neither survives YAML serialization. The pipeline ``input_schema``
    docstring states this explicitly; these tests pin both failure modes in
    code so that any future change to add round-trip support must update the
    contract intentionally (delete these tests, not flip them to
    ``assert success``).

    Related finding for the graph_artifact owner: ``_encode_init_inputs``
    silently passes through OPAQUE values that lack ``get_identifier`` (see
    the "Unknown opaque shape — defer to caller serialization at their own
    risk." branch), which is what lets pipelines reach the (broken)
    serialization path at all. A fail-loud guard there would raise a clearer
    error than either downstream failure mode below.
    """

    async def test_default_init_async_kwargs_mappingproxy_breaks_asdict(self, mock_objective_target):
        """PhaseSpec defaults ``init_async_kwargs`` to a MappingProxyType, which
        ``dataclasses.asdict`` cannot ``deepcopy``. ``graph_artifact_to_yaml``
        starts with ``asdict(artifact)`` so this is the first failure point.
        """
        from dataclasses import asdict

        from pyrit.scenario.core.graph_artifact import build_graph_artifact

        inner = _make_inner_scenario("p1")
        spec = PhaseSpec(name="p1", scenario_factory=lambda: inner)
        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(objective_target=mock_objective_target)

        artifact = build_graph_artifact(pipeline, init_inputs={"phases": [spec]})
        assert isinstance(artifact.init_inputs["phases"], list)
        assert isinstance(artifact.init_inputs["phases"][0], PhaseSpec)

        with pytest.raises(TypeError, match="mappingproxy"):
            asdict(artifact)

    async def test_callable_factory_breaks_yaml_dump(self, mock_objective_target, tmp_path):
        """Even when ``init_async_kwargs`` is supplied as a plain dict so
        ``asdict`` succeeds, YAML's ``SafeDumper`` cannot represent the
        Callable scenario factory.
        """
        import yaml

        from pyrit.scenario.core.graph_artifact import (
            build_graph_artifact,
            graph_artifact_to_yaml,
        )

        spec, _ = _phase_spec("p1")  # init_kwargs={} avoids the MappingProxy path
        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(objective_target=mock_objective_target)

        artifact = build_graph_artifact(pipeline, init_inputs={"phases": [spec]})
        out_path = tmp_path / "pipeline.yaml"
        with pytest.raises(yaml.representer.RepresenterError, match="cannot represent"):
            graph_artifact_to_yaml(artifact, out_path)


# --------------------------------------------------------------------------- #
# Metadata merge order (pipeline keys always win)
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("patch_central_database")
class TestPhaseActionMetadataMergeOrder:
    """Pin that pipeline-stamped metadata wins over inner step result metadata.

    The orchestrator's logging and ``step_identifier`` consumers downstream of
    ``_build_phase_action`` always need the pipeline's view of
    ``step_name`` / ``phase_index``, not whatever the inner step decided to
    stamp. A misbehaving inner that emits ``"step_name"`` in its result
    metadata must not override the pipeline's stamp.
    """

    async def test_pipeline_keys_override_inner_step_metadata(self, mock_objective_target):
        from pyrit.scenario.core.scenario_step import ScenarioStep, ScenarioStepResult

        class _NoisyStep(ScenarioStep):
            """A step whose process_async result carries colliding metadata keys."""

            def __init__(self, name: str) -> None:
                self.name = name
                self.outputs = ["done"]

            async def process_async(self) -> ScenarioStepResult:
                return ScenarioStepResult(
                    outcome="done",
                    attack_results=[],
                    metadata={
                        "step_name": "INNER_HIJACK",
                        "phase_index": 999,
                        "innocuous": "inner_value",
                    },
                )

            def _build_identifier(self) -> ComponentIdentifier:
                return ComponentIdentifier.of(self, params={"name": self.name})

        spec, _ = _phase_spec("pipeline_phase_zero")
        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(objective_target=mock_objective_target)

        noisy = _NoisyStep(name="pipeline_phase_zero")
        action = pipeline._build_phase_action(index=0, step=noisy)
        graph = pipeline._build_execution_graph(steps=[noisy])

        next_state, result = await action(graph)

        assert next_state == 1
        assert result is not None
        # Pipeline keys win:
        assert result.metadata["step_name"] == "pipeline_phase_zero"
        assert result.metadata["phase_index"] == 0
        # Innocuous inner keys still pass through:
        assert result.metadata["innocuous"] == "inner_value"


# --------------------------------------------------------------------------- #
# Persisted ScenarioResult.metadata["phase_executions"]
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("patch_central_database")
class TestPhaseExecutionsPersistence:
    """Pin that per-phase outcomes are written into the outer ScenarioResult.metadata.

    Without this persistence, cross-process readers of the pipeline's
    ``ScenarioResult`` (anything reloading via ``get_scenario_results``
    without holding a live pipeline reference) would see only the composition
    (empty phase buckets) and no per-phase outcome record. The persisted
    snapshot is a list of ``{"name", "outcome", "inner_scenario_result_id"}``
    dicts keyed under ``metadata["phase_executions"]``.
    """

    async def test_phase_executions_persisted_in_scenario_result_metadata(self, mock_objective_target):
        from pyrit.memory import CentralMemory

        spec_a, inner_a = _phase_spec("alpha")
        spec_b, inner_b = _phase_spec("beta")

        pipeline = ScenarioPipeline(phases=[spec_a, spec_b])
        await pipeline.initialize_async(objective_target=mock_objective_target)
        result = await pipeline.run_async()

        memory = CentralMemory.get_memory_instance()
        [persisted] = memory.get_scenario_results(scenario_result_ids=[str(result.id)])
        snapshot = persisted.metadata["phase_executions"]

        assert isinstance(snapshot, list)
        assert len(snapshot) == 2
        assert snapshot[0]["name"] == "alpha"
        assert snapshot[0]["outcome"] == _PHASE_OUTCOME_COMPLETED
        assert snapshot[0]["inner_scenario_result_id"] == str(inner_a.run_async.return_value.id)
        assert snapshot[1]["name"] == "beta"
        assert snapshot[1]["outcome"] == _PHASE_OUTCOME_COMPLETED
        assert snapshot[1]["inner_scenario_result_id"] == str(inner_b.run_async.return_value.id)

    async def test_skipped_phase_persists_with_null_inner_result_id(self, mock_objective_target):
        from pyrit.memory import CentralMemory

        spec_a, _ = _phase_spec("a")
        spec_b, inner_b = _conditional_phase_spec("b", skip_when=lambda _ctx: True)

        pipeline = ScenarioPipeline(phases=[spec_a, spec_b])
        await pipeline.initialize_async(objective_target=mock_objective_target)
        result = await pipeline.run_async()

        memory = CentralMemory.get_memory_instance()
        [persisted] = memory.get_scenario_results(scenario_result_ids=[str(result.id)])
        snapshot = persisted.metadata["phase_executions"]

        assert snapshot[1]["name"] == "b"
        assert snapshot[1]["outcome"] == _PHASE_OUTCOME_SKIPPED
        assert snapshot[1]["inner_scenario_result_id"] is None
        # Predicate elided execution, factory should never have been called.
        inner_b.run_async.assert_not_called()

    async def test_finalize_hook_preserves_existing_metadata_keys(self, mock_objective_target):
        """The finalize override merges with prior metadata, doesn't replace it."""
        from pyrit.memory import CentralMemory

        spec, _ = _phase_spec("only")
        pipeline = ScenarioPipeline(phases=[spec])
        await pipeline.initialize_async(objective_target=mock_objective_target)

        # Simulate a prior metadata entry (e.g. what the base _build_initial_scenario_metadata
        # would write when max_dataset_size is set).
        memory = CentralMemory.get_memory_instance()
        memory.update_scenario_metadata(
            scenario_result_id=str(pipeline._scenario_result_id),
            metadata={"objective_hashes": ["sentinel-pre-existing-hash"]},
        )

        result = await pipeline.run_async()
        [persisted] = memory.get_scenario_results(scenario_result_ids=[str(result.id)])

        # Both keys must coexist; the finalize hook merges, doesn't replace.
        assert persisted.metadata["objective_hashes"] == ["sentinel-pre-existing-hash"]
        assert "phase_executions" in persisted.metadata
        assert persisted.metadata["phase_executions"][0]["name"] == "only"
