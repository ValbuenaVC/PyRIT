# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the ``pyrit_wizard`` CLI module."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.cli import pyrit_wizard
from pyrit.common.parameter import Parameter
from pyrit.scenario.core.input_collector import OpaqueRoleNotElicitableError
from pyrit.scenario.core.input_schema import RoleDescriptor, RoleTag

# --- parse_args ----------------------------------------------------------------


def test_parse_args_defaults_for_bare_scenario_name():
    args = pyrit_wizard.parse_args(["my_scenario"])

    assert args.scenario_name == "my_scenario"
    assert args.list_scenarios is False
    assert args.list_targets is False
    assert args.target is None
    assert args.save is None
    assert args.run is False
    assert args.log_level == logging.WARNING
    assert args.max_attempts_per_role == 5
    assert args.non_interactive is False


def test_parse_args_list_scenarios_flag():
    args = pyrit_wizard.parse_args(["--list-scenarios"])

    assert args.list_scenarios is True
    assert args.scenario_name is None


def test_parse_args_full_command_line():
    args = pyrit_wizard.parse_args(
        [
            "my_scenario",
            "--target",
            "my_target",
            "--save",
            "out.yaml",
            "--run",
            "--initializers",
            "target",
            "--max-concurrency",
            "4",
            "--max-retries",
            "2",
            "--max-attempts-per-role",
            "3",
            "--non-interactive",
        ]
    )

    assert args.scenario_name == "my_scenario"
    assert args.target == "my_target"
    assert args.save == Path("out.yaml")
    assert args.run is True
    assert args.initializers == ["target"]
    assert args.max_concurrency == 4
    assert args.max_retries == 2
    assert args.max_attempts_per_role == 3
    assert args.non_interactive is True


# --- _role_from_parameter -------------------------------------------------------


def test_role_from_parameter_scalar_required_when_no_default():
    parameter = Parameter(name="threshold", description="cutoff", param_type=int)

    role = pyrit_wizard._role_from_parameter(parameter)

    assert role.name == "threshold"
    assert role.tag is RoleTag.SCALAR
    assert role.param_type is int
    assert role.required is True
    assert role.default is None


def test_role_from_parameter_optional_when_default_present():
    parameter = Parameter(name="batch_size", description="batch", default=10, param_type=int)

    role = pyrit_wizard._role_from_parameter(parameter)

    assert role.required is False
    assert role.default == 10


def test_role_from_parameter_choice_when_choices_declared():
    parameter = Parameter(
        name="mode",
        description="execution mode",
        default="fast",
        param_type=str,
        choices=("fast", "thorough"),
    )

    role = pyrit_wizard._role_from_parameter(parameter)

    assert role.tag is RoleTag.CHOICE
    assert role.choices == ("fast", "thorough")
    assert role.required is False


# --- _pick_from_menu ------------------------------------------------------------


def test_pick_from_menu_accepts_index():
    with patch("builtins.input", return_value="2"):
        choice = pyrit_wizard._pick_from_menu(prompt="pick:", options=["a", "b", "c"])

    assert choice == "b"


def test_pick_from_menu_accepts_exact_name():
    with patch("builtins.input", return_value="c"):
        choice = pyrit_wizard._pick_from_menu(prompt="pick:", options=["a", "b", "c"])

    assert choice == "c"


def test_pick_from_menu_reprompts_on_invalid_then_succeeds(capsys):
    inputs = iter(["foo", "9", "1"])
    with patch("builtins.input", side_effect=lambda _: next(inputs)):
        choice = pyrit_wizard._pick_from_menu(prompt="pick:", options=["alpha", "beta"])

    assert choice == "alpha"
    captured = capsys.readouterr()
    assert "Invalid input" in captured.out
    assert "out of range" in captured.out


def test_pick_from_menu_raises_on_empty_options():
    with pytest.raises(ValueError, match="empty"):
        pyrit_wizard._pick_from_menu(prompt="pick:", options=[])


# --- _resolve_scenario_class ----------------------------------------------------


def _make_context_with_scenarios(scenarios: dict[str, type]) -> MagicMock:
    registry = MagicMock()
    registry.get_names.return_value = list(scenarios)
    registry.get_class.side_effect = lambda name: scenarios[name]
    context = MagicMock()
    context.scenario_registry = registry
    return context


def test_resolve_scenario_class_uses_explicit_name():
    fake_cls = type("FakeScenario", (), {})
    context = _make_context_with_scenarios({"my_scenario": fake_cls})

    cls, name = pyrit_wizard._resolve_scenario_class(context=context, explicit_name="my_scenario", interactive=True)

    assert cls is fake_cls
    assert name == "my_scenario"


def test_resolve_scenario_class_raises_when_explicit_missing():
    fake_cls = type("Foo", (), {})
    context = _make_context_with_scenarios({"foo": fake_cls})
    context.scenario_registry.get_class.side_effect = KeyError("bar")

    with pytest.raises(ValueError, match="not found"):
        pyrit_wizard._resolve_scenario_class(context=context, explicit_name="bar", interactive=True)


def test_resolve_scenario_class_interactive_pick():
    cls_a = type("A", (), {})
    cls_b = type("B", (), {})
    context = _make_context_with_scenarios({"alpha": cls_a, "beta": cls_b})

    with patch("builtins.input", return_value="beta"):
        cls, name = pyrit_wizard._resolve_scenario_class(context=context, explicit_name=None, interactive=True)

    assert cls is cls_b
    assert name == "beta"


def test_resolve_scenario_class_non_interactive_no_name_raises():
    context = _make_context_with_scenarios({"foo": type("Foo", (), {})})

    with pytest.raises(ValueError, match="non-interactive"):
        pyrit_wizard._resolve_scenario_class(context=context, explicit_name=None, interactive=False)


def test_resolve_scenario_class_raises_when_registry_empty():
    context = _make_context_with_scenarios({})

    with pytest.raises(RuntimeError, match="No scenarios are registered"):
        pyrit_wizard._resolve_scenario_class(context=context, explicit_name=None, interactive=True)


# --- _resolve_target_name -------------------------------------------------------


def test_resolve_target_name_uses_explicit_name():
    name = pyrit_wizard._resolve_target_name(explicit_name="t1", available_targets=["t1", "t2"], interactive=True)

    assert name == "t1"


def test_resolve_target_name_rejects_unknown_explicit_name():
    with pytest.raises(ValueError, match="not found"):
        pyrit_wizard._resolve_target_name(explicit_name="t9", available_targets=["t1", "t2"], interactive=True)


def test_resolve_target_name_interactive_picker():
    with patch("builtins.input", return_value="2"):
        name = pyrit_wizard._resolve_target_name(explicit_name=None, available_targets=["one", "two"], interactive=True)

    assert name == "two"


def test_resolve_target_name_non_interactive_without_target_raises():
    with pytest.raises(ValueError, match="--non-interactive"):
        pyrit_wizard._resolve_target_name(explicit_name=None, available_targets=["t1"], interactive=False)


def test_resolve_target_name_empty_registry_raises():
    with pytest.raises(RuntimeError, match="No targets are registered"):
        pyrit_wizard._resolve_target_name(explicit_name=None, available_targets=[], interactive=True)


# --- _resolve_strategies --------------------------------------------------------


class _FakeStrategy:
    """Minimal stand-in for a ScenarioStrategy enum member."""

    def __init__(self, value: str) -> None:
        self.value = value
        self.name = value.upper()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeStrategy) and other.value == self.value

    def __hash__(self) -> int:
        return hash(self.value)


class _FakeStrategyClass:
    """Iterable+callable stand-in for a ScenarioStrategy enum class."""

    def __init__(self, values: list[str]) -> None:
        self._members = [_FakeStrategy(v) for v in values]

    def __iter__(self):
        return iter(self._members)

    def __call__(self, value: str) -> _FakeStrategy:
        for member in self._members:
            if member.value == value:
                return member
        raise ValueError(value)


def _make_strategy_class(values: list[str]) -> _FakeStrategyClass:
    return _FakeStrategyClass(values)


def test_resolve_strategies_returns_none_on_blank_input():
    scenario_class = MagicMock()
    scenario_class.__name__ = "Demo"
    scenario_class.get_strategy_class.return_value = _make_strategy_class(["a", "b"])

    with patch("builtins.input", return_value=""):
        result = pyrit_wizard._resolve_strategies(scenario_class=scenario_class, interactive=True)

    assert result is None


def test_resolve_strategies_returns_none_in_non_interactive_mode():
    scenario_class = MagicMock()
    scenario_class.__name__ = "Demo"

    result = pyrit_wizard._resolve_strategies(scenario_class=scenario_class, interactive=False)

    assert result is None
    scenario_class.get_strategy_class.assert_not_called()


def test_resolve_strategies_picks_by_index_and_name():
    scenario_class = MagicMock()
    scenario_class.__name__ = "Demo"
    scenario_class.get_strategy_class.return_value = _make_strategy_class(["alpha", "beta", "gamma"])

    with patch("builtins.input", return_value="1,gamma"):
        result = pyrit_wizard._resolve_strategies(scenario_class=scenario_class, interactive=True)

    assert result is not None
    assert [member.value for member in result] == ["alpha", "gamma"]


def test_resolve_strategies_raises_on_unknown_name():
    scenario_class = MagicMock()
    scenario_class.__name__ = "Demo"
    scenario_class.get_strategy_class.return_value = _make_strategy_class(["alpha"])

    with patch("builtins.input", return_value="bogus"):
        with pytest.raises(ValueError, match="not recognized"):
            pyrit_wizard._resolve_strategies(scenario_class=scenario_class, interactive=True)


def test_resolve_strategies_raises_on_out_of_range_index():
    scenario_class = MagicMock()
    scenario_class.__name__ = "Demo"
    scenario_class.get_strategy_class.return_value = _make_strategy_class(["alpha"])

    with patch("builtins.input", return_value="9"):
        with pytest.raises(ValueError, match="out of range"):
            pyrit_wizard._resolve_strategies(scenario_class=scenario_class, interactive=True)


# --- _collect_init_inputs -------------------------------------------------------


def test_collect_init_inputs_returns_empty_for_default_schema():
    scenario_class = MagicMock()
    scenario_class.input_schema.return_value = []

    result = pyrit_wizard._collect_init_inputs(scenario_class=scenario_class, interactive=True, max_attempts=5)

    assert result == {}


def test_collect_init_inputs_surfaces_opaque_role_with_pointer_to_artifact():
    scenario_class = MagicMock()
    scenario_class.__name__ = "OpaqueScenario"
    opaque_role = RoleDescriptor(
        name="big_obj",
        description="pre-built thing",
        tag=RoleTag.OPAQUE,
        required=True,
    )
    scenario_class.input_schema.return_value = [opaque_role]

    fake_collector = MagicMock()
    fake_collector.collect.side_effect = OpaqueRoleNotElicitableError("big_obj")
    with patch(
        "pyrit.scenario.core.input_collector.CliInputCollector",
        return_value=fake_collector,
    ):
        with pytest.raises(RuntimeError, match="--from-artifact"):
            pyrit_wizard._collect_init_inputs(
                scenario_class=scenario_class,
                interactive=True,
                max_attempts=5,
            )


def test_collect_init_inputs_non_interactive_with_schema_raises():
    scenario_class = MagicMock()
    scenario_class.__name__ = "ConfiguredScenario"
    scenario_class.input_schema.return_value = [
        RoleDescriptor(name="x", description="x", tag=RoleTag.SCALAR, param_type=int)
    ]

    with pytest.raises(ValueError, match="non-interactive"):
        pyrit_wizard._collect_init_inputs(scenario_class=scenario_class, interactive=False, max_attempts=5)


def test_collect_init_inputs_returns_collected_dict():
    scenario_class = MagicMock()
    scenario_class.__name__ = "Demo"
    role = RoleDescriptor(name="alpha", description="a", tag=RoleTag.SCALAR, param_type=str)
    scenario_class.input_schema.return_value = [role]

    fake_collector = MagicMock()
    fake_collector.collect.return_value = "yes"

    with patch(
        "pyrit.scenario.core.input_collector.CliInputCollector",
        return_value=fake_collector,
    ):
        result = pyrit_wizard._collect_init_inputs(scenario_class=scenario_class, interactive=True, max_attempts=5)

    assert result == {"alpha": "yes"}


# --- _collect_init_async_inputs -------------------------------------------------


def test_collect_init_async_inputs_no_params_returns_overrides_only():
    scenario_class = MagicMock()
    scenario_class.supported_parameters.return_value = []

    result = pyrit_wizard._collect_init_async_inputs(
        scenario_class=scenario_class,
        interactive=True,
        max_attempts=5,
        cli_overrides={"max_concurrency": 4, "max_retries": None},
    )

    assert result == {"max_concurrency": 4}


def test_collect_init_async_inputs_overrides_win_over_prompts():
    scenario_class = MagicMock()
    scenario_class.__name__ = "Demo"
    scenario_class.supported_parameters.return_value = [
        Parameter(name="threshold", description="t", default=1, param_type=int)
    ]

    fake_collector = MagicMock()
    fake_collector.collect.return_value = 5

    with patch(
        "pyrit.scenario.core.input_collector.CliInputCollector",
        return_value=fake_collector,
    ):
        result = pyrit_wizard._collect_init_async_inputs(
            scenario_class=scenario_class,
            interactive=True,
            max_attempts=5,
            cli_overrides={"threshold": 99, "max_concurrency": None},
        )

    assert result["threshold"] == 99


# --- _print_scenarios_list ------------------------------------------------------


def test_print_scenarios_list_empty(capsys):
    registry = MagicMock()
    registry.list_metadata.return_value = []
    context = MagicMock()
    context.scenario_registry = registry

    rc = pyrit_wizard._print_scenarios_list(context=context)
    captured = capsys.readouterr()

    assert rc == 0
    assert "No scenarios are registered." in captured.out


def test_print_scenarios_list_renders_first_description_line(capsys):
    registry = MagicMock()
    item = MagicMock()
    item.registry_name = "foo"
    item.class_description = "First line.\nSecond line should be hidden."
    registry.list_metadata.return_value = [item]
    context = MagicMock()
    context.scenario_registry = registry

    rc = pyrit_wizard._print_scenarios_list(context=context)
    captured = capsys.readouterr()

    assert rc == 0
    assert "foo — First line." in captured.out
    assert "Second line should be hidden." not in captured.out


# --- _print_targets_list --------------------------------------------------------


def test_print_targets_list_empty(capsys):
    rc = pyrit_wizard._print_targets_list(target_names=[])
    captured = capsys.readouterr()

    assert rc == 0
    assert "No targets are registered." in captured.out


def test_print_targets_list_lists_names(capsys):
    rc = pyrit_wizard._print_targets_list(target_names=["t1", "t2"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Registered targets (2)" in captured.out
    assert "t1" in captured.out
    assert "t2" in captured.out


# --- main / _run_wizard_async ---------------------------------------------------


def test_main_handles_keyboard_interrupt(capsys):
    with patch(
        "pyrit.cli.pyrit_wizard._run_wizard_async",
        new=AsyncMock(side_effect=KeyboardInterrupt()),
    ):
        rc = pyrit_wizard.main(["--list-scenarios"])
    captured = capsys.readouterr()

    assert rc == 130
    assert "Aborted by user." in captured.out


def test_main_returns_one_on_unexpected_exception(capsys):
    with patch(
        "pyrit.cli.pyrit_wizard._run_wizard_async",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        rc = pyrit_wizard.main(["--list-scenarios"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "Error: boom" in captured.out


def test_main_returns_zero_on_success():
    with patch("pyrit.cli.pyrit_wizard._run_wizard_async", new=AsyncMock(return_value=0)):
        rc = pyrit_wizard.main(["--list-scenarios"])

    assert rc == 0


async def test_run_wizard_async_list_scenarios_short_circuits(capsys):
    """--list-scenarios returns before any target / scenario resolution runs."""
    with patch("pyrit.cli.frontend_core.FrontendCore") as fake_frontend_cls:
        context = MagicMock()
        context.initialize_async = AsyncMock()
        registry = MagicMock()
        item = MagicMock()
        item.registry_name = "foo"
        item.class_description = "Foo scenario"
        registry.list_metadata.return_value = [item]
        context.scenario_registry = registry
        fake_frontend_cls.return_value = context

        parsed = pyrit_wizard.parse_args(["--list-scenarios"])
        rc = await pyrit_wizard._run_wizard_async(parsed_args=parsed)

    captured = capsys.readouterr()
    assert rc == 0
    assert "foo — Foo scenario" in captured.out
    # No target listing or scenario picker triggered.
    assert "Available targets" not in captured.out
    assert "Wizard:" not in captured.out


async def test_run_wizard_async_initialization_script_missing_returns_one(capsys):
    with patch(
        "pyrit.cli.frontend_core.resolve_initialization_scripts",
        side_effect=FileNotFoundError("nope.py not found"),
    ):
        parsed = pyrit_wizard.parse_args(["--list-scenarios", "--initialization-scripts", "nope.py"])
        rc = await pyrit_wizard._run_wizard_async(parsed_args=parsed)

    captured = capsys.readouterr()
    assert rc == 1
    assert "nope.py not found" in captured.out
