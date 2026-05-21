# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
PyRIT CLI - Interactive wizard for assembling a scenario from declared inputs.

The wizard composes the Phase 8a/8b/8c/8g surfaces:

* :func:`pyrit.scenario.core.builder.discover_input_schema` —
  rich-object ``__init__`` roles declared by the scenario class.
* :func:`pyrit.scenario.core.builder.discover_supported_parameters` —
  scalar ``initialize_async`` parameters (legacy contract).
* :class:`pyrit.scenario.core.input_collector.CliInputCollector` —
  stdin/stdout elicitation with the error-recovery retry loop.
* :func:`pyrit.scenario.core.builder.build_scenario_from_inputs` —
  the single entry point that constructs + initializes the scenario.
* :func:`pyrit.scenario.core.graph_artifact.build_graph_artifact` /
  :func:`pyrit.scenario.core.graph_artifact.graph_artifact_to_yaml` —
  optional ``--save artifact.yaml`` capture.

Use ``pyrit_scan --from-artifact path.yaml`` (Phase 8e) to replay an artifact.

Examples::

  # List available scenarios and exit
  pyrit_wizard --list-scenarios

  # Pick a scenario interactively, prompting for every input
  pyrit_wizard --target my_target --initializers target

  # Skip the interactive picker, save an artifact, but don't run yet
  pyrit_wizard foundry.red_team_agent --target my_target --save my_run.yaml \\
      --initializers target

  # Build + run end-to-end with one elicitation pass
  pyrit_wizard garak.encoding --target my_target --run --initializers target
"""

from __future__ import annotations

import asyncio
import logging
import sys
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pyrit.cli._cli_args import (
    ARG_HELP,
    _parse_initializer_arg,
    non_negative_int,
    positive_int,
    validate_log_level_argparse,
)

if TYPE_CHECKING:
    from pyrit.cli.frontend_core import FrontendCore
    from pyrit.common.parameter import Parameter
    from pyrit.scenario.core import Scenario
    from pyrit.scenario.core.input_schema import RoleDescriptor


_DESCRIPTION = """PyRIT Wizard - Interactively assemble and (optionally) run a scenario.

Walks the scenario's declared input_schema() (rich __init__ roles) and
supported_parameters() (scalar initialize_async args) via an interactive
prompt, then builds the scenario, optionally saves a reproducible artifact,
and optionally runs it.

Examples:
  # List available scenarios
  pyrit_wizard --list-scenarios

  # Wizard for a specific scenario, save artifact, don't run
  pyrit_wizard foundry.red_team_agent --target my_target --save out.yaml \\
      --initializers target

  # Build + run end-to-end
  pyrit_wizard garak.encoding --target my_target --run --initializers target

Note: Scenarios whose input_schema() declares OPAQUE roles (e.g. pre-built
AtomicAttack or OutcomeScorer instances) cannot be elicited end-to-end from a
CLI. The wizard exits with a helpful pointer to the programmatic API or
``pyrit_scan --from-artifact path.yaml``.
"""


def _build_parser() -> ArgumentParser:
    """
    Build the ``pyrit_wizard`` argparse parser.

    Returns:
        ArgumentParser: The fully configured argument parser.
    """
    parser = ArgumentParser(
        prog="pyrit_wizard",
        description=_DESCRIPTION,
        formatter_class=RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "scenario_name",
        type=str,
        nargs="?",
        help="Name of the scenario to wizard. If omitted, you'll pick from a menu.",
    )

    parser.add_argument(
        "--config-file",
        type=Path,
        help=ARG_HELP["config_file"],
    )

    parser.add_argument(
        "--log-level",
        type=validate_log_level_argparse,
        default=logging.WARNING,
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: WARNING)",
    )

    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List all available scenarios and exit",
    )

    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="List registered targets and exit. Requires initializers that register targets "
        "(e.g., --initializers target).",
    )

    parser.add_argument(
        "--initializers",
        type=_parse_initializer_arg,
        nargs="+",
        help=ARG_HELP["initializers"],
    )

    parser.add_argument(
        "--initialization-scripts",
        type=str,
        nargs="+",
        help=ARG_HELP["initialization_scripts"],
    )

    parser.add_argument(
        "--target",
        type=str,
        help="Name of a registered target. If omitted, the wizard prompts interactively.",
    )

    parser.add_argument(
        "--save",
        type=Path,
        help="Write the built scenario as a graph artifact (YAML) to this path.",
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the built scenario after elicitation completes.",
    )

    parser.add_argument(
        "--max-concurrency",
        type=positive_int,
        help=ARG_HELP["max_concurrency"],
    )

    parser.add_argument(
        "--max-retries",
        type=non_negative_int,
        help=ARG_HELP["max_retries"],
    )

    parser.add_argument(
        "--max-attempts-per-role",
        type=positive_int,
        default=5,
        help="Maximum re-prompt attempts for any single role before bailing (default: 5).",
    )

    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive prompts; fail fast on any missing input. "
        "Useful for scripted runs that supply every input via --config-file.",
    )

    return parser


def parse_args(args: Optional[list[str]] = None) -> Namespace:
    """
    Parse command-line arguments for ``pyrit_wizard``.

    Args:
        args: Optional argv list (mainly for tests). ``None`` means use ``sys.argv``.

    Returns:
        Namespace: The parsed argument namespace.
    """
    return _build_parser().parse_args(args)


def _role_from_parameter(parameter: Parameter) -> RoleDescriptor:
    """
    Adapt a :class:`Parameter` (scalar ``initialize_async`` declaration) to a
    :class:`RoleDescriptor` so the same :class:`CliInputCollector` retry loop
    can elicit both halves of the lifecycle.

    Choice-typed parameters become :attr:`RoleTag.CHOICE`; everything else is
    treated as a :attr:`RoleTag.SCALAR`. Optionality follows the parameter's
    default: a non-None default makes the role optional.

    The collector treats a role as required iff ``required=True``, so we set
    ``required=False`` whenever the parameter declares a default (the
    scenario's ``set_params_from_args`` will materialize the default if the
    user skips the prompt).

    Args:
        parameter: The :class:`Parameter` declared by the scenario.

    Returns:
        RoleDescriptor: The adapted descriptor consumed by :class:`CliInputCollector`.
    """
    from pyrit.scenario.core.input_schema import RoleDescriptor, RoleTag

    tag = RoleTag.CHOICE if parameter.choices is not None else RoleTag.SCALAR
    required = parameter.default is None and parameter.choices is None
    return RoleDescriptor(
        name=parameter.name,
        description=parameter.description,
        tag=tag,
        param_type=parameter.param_type,
        choices=parameter.choices,
        default=parameter.default,
        required=required,
    )


def _pick_from_menu(*, prompt: str, options: list[str]) -> str:
    """
    Show a numbered menu of ``options`` and return the chosen item.

    Accepts either a 1-based index or an exact item name. Re-prompts on
    invalid input until the user provides a recognized value or EOFs.

    Args:
        prompt: Prompt text shown above the option list.
        options: Available choices to display, in order.

    Returns:
        str: The chosen option.

    Raises:
        ValueError: When ``options`` is empty.
    """
    if not options:
        raise ValueError("Cannot pick from an empty option list.")

    while True:
        print(prompt)
        for index, option in enumerate(options, start=1):
            print(f"  {index}. {option}")
        raw = input("> ").strip()
        if raw in options:
            return raw
        try:
            picked = int(raw)
        except ValueError:
            print(f"  ! Invalid input {raw!r}; enter a number 1-{len(options)} or an item name.")
            continue
        if 1 <= picked <= len(options):
            return options[picked - 1]
        print(f"  ! Index {picked} out of range; enter a number 1-{len(options)}.")


def _resolve_scenario_class(
    *,
    context: FrontendCore,
    explicit_name: str | None,
    interactive: bool,
) -> tuple[type[Scenario], str]:
    """
    Resolve the scenario class to wizard.

    Returns the class and the registry name it was found under so callers can
    print user-friendly progress messages. When ``explicit_name`` is provided
    we look it up directly; otherwise (when ``interactive=True``) we offer a
    numbered menu of registered scenarios.

    Args:
        context: The initialized :class:`FrontendCore` (registry source).
        explicit_name: Scenario name supplied via positional arg, if any.
        interactive: When True, the wizard may prompt for missing inputs.

    Returns:
        tuple[type[Scenario], str]: Resolved scenario class and the registry
            name it was looked up under.

    Raises:
        RuntimeError: When the scenario registry is empty.
        ValueError: When ``explicit_name`` is unknown, or when no name was
            supplied and ``interactive`` is False.
    """
    registry = context.scenario_registry
    names = sorted(registry.get_names())
    if not names:
        raise RuntimeError("No scenarios are registered. Did you forget --initializers or --initialization-scripts?")

    if explicit_name:
        try:
            scenario_class = registry.get_class(explicit_name)
        except KeyError as exc:
            raise ValueError(f"Scenario {explicit_name!r} not found. Available: {', '.join(names)}") from exc
        return scenario_class, explicit_name

    if not interactive:
        raise ValueError(
            "No scenario_name provided and --non-interactive was set. "
            "Either pass a positional scenario name or drop --non-interactive."
        )

    chosen = _pick_from_menu(prompt="Available scenarios:", options=names)
    return registry.get_class(chosen), chosen


def _resolve_target_name(
    *,
    explicit_name: str | None,
    available_targets: list[str],
    interactive: bool,
) -> str:
    """
    Resolve the objective_target name, prompting interactively when needed.

    Args:
        explicit_name: Target name supplied via ``--target``, if any.
        available_targets: Registered target names from the populated
            :class:`TargetRegistry`.
        interactive: When True, the wizard may prompt the user.

    Returns:
        str: The resolved target name.

    Raises:
        RuntimeError: When no targets are registered.
        ValueError: When ``explicit_name`` is unknown, or when no target was
            supplied and ``interactive`` is False.
    """
    if explicit_name is not None:
        if explicit_name not in available_targets:
            raise ValueError(
                f"Target {explicit_name!r} not found. Available: {', '.join(available_targets) or '(none)'}"
            )
        return explicit_name

    if not available_targets:
        raise RuntimeError(
            "No targets are registered. Did you forget --initializers target "
            "or --initialization-scripts pointing at a script that registers a target?"
        )

    if not interactive:
        raise ValueError("No --target provided and --non-interactive was set. Pass --target NAME explicitly.")

    return _pick_from_menu(prompt="Available targets:", options=available_targets)


def _resolve_strategies(
    *,
    scenario_class: type[Scenario],
    interactive: bool,
) -> list[Any] | None:
    """
    Optionally elicit scenario strategies from the user.

    Returns ``None`` when the user skips the prompt (the scenario then uses
    its declared default aggregate). Returns a non-empty list of enum members
    when the user supplied at least one selection.

    Args:
        scenario_class: The scenario whose strategy enum should be elicited.
        interactive: When False, the wizard skips strategy elicitation entirely.

    Returns:
        list[Any] | None: Selected enum members, or ``None`` if the user skipped
            the prompt or the scenario has no declared strategy enum.

    Raises:
        ValueError: When a supplied strategy name or index is invalid.
    """
    if not interactive:
        return None

    try:
        strategy_class = scenario_class.get_strategy_class()
    except Exception:
        return None

    members = [member.value for member in strategy_class]
    if not members:
        return None

    print(
        f"\nScenario strategies for {scenario_class.__name__} (comma-separated names or numbers, blank = use defaults):"
    )
    for index, member in enumerate(members, start=1):
        print(f"  {index}. {member}")
    raw = input("> ").strip()
    if not raw:
        return None

    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    selected: list[Any] = []
    for token in tokens:
        if token in members:
            selected.append(strategy_class(token))
            continue
        try:
            index = int(token)
        except ValueError as exc:
            raise ValueError(f"Strategy {token!r} not recognized. Available: {', '.join(members)}") from exc
        if not 1 <= index <= len(members):
            raise ValueError(f"Strategy index {index} out of range (1-{len(members)}).")
        selected.append(strategy_class(members[index - 1]))

    return selected or None


def _collect_init_inputs(
    *,
    scenario_class: type[Scenario],
    interactive: bool,
    max_attempts: int,
) -> dict[str, Any]:
    """
    Walk the scenario's ``input_schema()`` and collect rich-object init inputs.

    OPAQUE roles surface a clear error pointing at the artifact-replay
    workflow — the wizard cannot elicit pre-built ``Identifiable`` instances
    from stdin alone.

    Args:
        scenario_class: The concrete scenario being constructed.
        interactive: When False, declared schema items raise instead of prompting.
        max_attempts: Per-role retry budget before bailing with
            :class:`MaxAttemptsExceededError`.

    Returns:
        dict[str, Any]: Collected values keyed by role name.

    Raises:
        RuntimeError: When the schema includes an OPAQUE role the CLI cannot elicit.
        ValueError: When the schema is non-empty but ``interactive`` is False.
    """
    from pyrit.scenario.core.builder import discover_input_schema
    from pyrit.scenario.core.input_collector import (
        CliInputCollector,
        OpaqueRoleNotElicitableError,
        collect_inputs_with_retry,
    )

    schema = discover_input_schema(scenario_class)
    if not schema:
        return {}

    if not interactive:
        raise ValueError(
            f"{scenario_class.__name__}.input_schema() declares {len(schema)} role(s) "
            "but --non-interactive was set. Use the programmatic API "
            "(build_scenario_from_inputs) or --config-file to supply them."
        )

    print(f"\n-- Constructor inputs for {scenario_class.__name__} --")
    collector = CliInputCollector()
    try:
        return collect_inputs_with_retry(
            collector=collector,
            schema=schema,
            max_attempts=max_attempts,
        )
    except OpaqueRoleNotElicitableError as exc:
        raise RuntimeError(
            f"{scenario_class.__name__} declares opaque constructor inputs that cannot be "
            f"elicited from the CLI: {exc}\n"
            "Build it programmatically via `build_scenario_from_inputs(...)`, or use a "
            "previously saved artifact via `pyrit_scan --from-artifact path.yaml`."
        ) from exc


def _collect_init_async_inputs(
    *,
    scenario_class: type[Scenario],
    interactive: bool,
    max_attempts: int,
    cli_overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Walk the scenario's ``supported_parameters()`` and collect init-async inputs.

    Adapts each :class:`Parameter` to a :class:`RoleDescriptor` and runs the
    same retry loop as constructor inputs. CLI overrides (``--max-concurrency``,
    ``--max-retries``) win over any elicited value.

    Args:
        scenario_class: The concrete scenario being initialized.
        interactive: When False, the wizard skips schema-driven prompting and
            relies on ``cli_overrides`` plus the scenario's own defaults.
        max_attempts: Per-role retry budget before bailing.
        cli_overrides: Values supplied on the command line. ``None`` entries
            are ignored; non-``None`` entries always win over prompts.

    Returns:
        dict[str, Any]: Collected ``initialize_async`` kwargs.
    """
    from pyrit.scenario.core.builder import discover_supported_parameters
    from pyrit.scenario.core.input_collector import (
        CliInputCollector,
        collect_inputs_with_retry,
    )

    parameters = discover_supported_parameters(scenario_class)
    if not parameters:
        collected: dict[str, Any] = {}
    elif not interactive:
        # In non-interactive mode the user is expected to supply everything via
        # CLI overrides (or accept declared defaults via set_params_from_args).
        collected = {}
    else:
        print(f"\n-- Runtime parameters for {scenario_class.__name__} --")
        schema = [_role_from_parameter(parameter) for parameter in parameters]
        collected = collect_inputs_with_retry(
            collector=CliInputCollector(),
            schema=schema,
            max_attempts=max_attempts,
        )

    # CLI overrides land on top so flags always win over prompts.
    collected.update({key: value for key, value in cli_overrides.items() if value is not None})
    return collected


def _print_scenarios_list(*, context: FrontendCore) -> int:
    """
    Print a one-line summary of every registered scenario and exit.

    Args:
        context: The initialized :class:`FrontendCore`.

    Returns:
        int: Process exit code (always ``0``).
    """
    registry = context.scenario_registry
    metadata_list = sorted(registry.list_metadata(), key=lambda m: m.registry_name)
    if not metadata_list:
        print("No scenarios are registered.")
        return 0
    print(f"Registered scenarios ({len(metadata_list)}):")
    for metadata in metadata_list:
        description = metadata.class_description.splitlines()[0] if metadata.class_description else ""
        print(f"  {metadata.registry_name}{f' — {description}' if description else ''}")
    return 0


def _print_targets_list(*, target_names: list[str]) -> int:
    """
    Print the list of registered targets and exit.

    Args:
        target_names: Names returned by :func:`frontend_core.list_targets_async`.

    Returns:
        int: Process exit code (always ``0``).
    """
    if not target_names:
        print(
            "No targets are registered. Pass --initializers target (or another "
            "initializer that registers targets) and try again."
        )
        return 0
    print(f"Registered targets ({len(target_names)}):")
    for name in target_names:
        print(f"  {name}")
    return 0


async def _run_wizard_async(*, parsed_args: Namespace) -> int:
    """
    Drive the wizard end-to-end.

    Returns an integer exit code suitable for ``sys.exit``.

    Args:
        parsed_args: The parsed argparse namespace.

    Returns:
        int: ``0`` on success, ``1`` on any handled error surfaced to stdout.

    Raises:
        RuntimeError: When the target registry drops the selected target between
            elicitation and build (defensive — indicates a fixture race).
    """
    # Deferred imports so ``--help`` stays instant.
    from pyrit.cli import frontend_core
    from pyrit.scenario.core.builder import build_scenario_from_inputs
    from pyrit.scenario.core.graph_artifact import (
        build_graph_artifact,
        graph_artifact_to_yaml,
    )

    interactive = not parsed_args.non_interactive

    initialization_scripts = None
    if parsed_args.initialization_scripts:
        try:
            initialization_scripts = frontend_core.resolve_initialization_scripts(
                script_paths=parsed_args.initialization_scripts
            )
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return 1

    context = frontend_core.FrontendCore(
        config_file=parsed_args.config_file,
        initialization_scripts=initialization_scripts,
        initializer_names=parsed_args.initializers,
        log_level=parsed_args.log_level,
    )
    await context.initialize_async()

    if parsed_args.list_scenarios:
        return _print_scenarios_list(context=context)

    # Populate the target registry via the same initializer flow pyrit_scan uses.
    target_names = await frontend_core.list_targets_async(context=context)

    if parsed_args.list_targets:
        return _print_targets_list(target_names=target_names)

    scenario_class, scenario_name = _resolve_scenario_class(
        context=context,
        explicit_name=parsed_args.scenario_name,
        interactive=interactive,
    )
    print(f"\nWizard: {scenario_name} ({scenario_class.__name__})")

    target_name = _resolve_target_name(
        explicit_name=parsed_args.target,
        available_targets=target_names,
        interactive=interactive,
    )

    init_inputs = _collect_init_inputs(
        scenario_class=scenario_class,
        interactive=interactive,
        max_attempts=parsed_args.max_attempts_per_role,
    )

    cli_overrides: dict[str, Any] = {
        "max_concurrency": parsed_args.max_concurrency,
        "max_retries": parsed_args.max_retries,
    }
    init_async_inputs = _collect_init_async_inputs(
        scenario_class=scenario_class,
        interactive=interactive,
        max_attempts=parsed_args.max_attempts_per_role,
        cli_overrides=cli_overrides,
    )

    # objective_target / scenario_strategies are injected after collection so
    # they always reflect explicit CLI or interactive picker choices, not a
    # value the user may have typed at a generic "objective_target" prompt.
    from pyrit.registry import TargetRegistry

    target_registry = TargetRegistry.get_registry_singleton()
    objective_target = target_registry.get_instance_by_name(target_name)
    if objective_target is None:
        # Defensive: list_targets_async populated the registry, so absence here
        # means a race with another test fixture clearing it. Surface a clear
        # message rather than letting build raise a TypeError downstream.
        raise RuntimeError(
            f"Target {target_name!r} disappeared from registry between elicitation and build. Re-run the wizard."
        )
    init_async_inputs["objective_target"] = objective_target

    strategies = _resolve_strategies(
        scenario_class=scenario_class,
        interactive=interactive,
    )
    if strategies is not None:
        init_async_inputs["scenario_strategies"] = strategies

    print(f"\nBuilding {scenario_class.__name__}...")
    sys.stdout.flush()
    scenario = await build_scenario_from_inputs(
        scenario_class,
        init_inputs=init_inputs,
        init_async_inputs=init_async_inputs,
    )
    print("  ok.")

    if parsed_args.save is not None:
        artifact = build_graph_artifact(
            scenario,
            init_inputs=init_inputs,
            init_async_inputs=init_async_inputs,
        )
        graph_artifact_to_yaml(artifact, parsed_args.save)
        print(f"  artifact saved -> {parsed_args.save}")

    if parsed_args.run:
        print(f"\nRunning {scenario_class.__name__}...")
        sys.stdout.flush()
        result = await scenario.run_async()
        from pyrit.output.scenario_result.pretty import PrettyScenarioResultMemoryPrinter

        printer = PrettyScenarioResultMemoryPrinter()
        await printer.print_summary_async(result)
    else:
        print(
            "\nDone. Pass --run to execute, or rerun with --save PATH and replay via `pyrit_scan --from-artifact PATH`."
        )

    return 0


def main(args: Optional[list[str]] = None) -> int:
    """
    Start the PyRIT wizard CLI.

    Returns:
        int: Exit code (0 on success, 1 on error).
    """
    try:
        parsed_args = parse_args(args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    print("Starting PyRIT wizard...")
    sys.stdout.flush()

    try:
        return asyncio.run(_run_wizard_async(parsed_args=parsed_args))
    except KeyboardInterrupt:
        print("\nAborted by user.")
        return 130
    except Exception as exc:
        print(f"\nError: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
