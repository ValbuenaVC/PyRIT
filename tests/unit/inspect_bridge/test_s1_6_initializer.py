# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Unit tests for S1.6 — InspectInitializer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_inspect_initializer_is_pyrit_initializer() -> None:
    from pyrit.inspect_bridge import InspectInitializer
    from pyrit.setup.initializers.pyrit_initializer import PyRITInitializer

    assert issubclass(InspectInitializer, PyRITInitializer)


def test_inspect_initializer_constructs() -> None:
    from pyrit.inspect_bridge import InspectInitializer

    init = InspectInitializer()
    assert init is not None


def test_inspect_initializer_log_dir_param() -> None:
    from pyrit.inspect_bridge import InspectInitializer

    init = InspectInitializer(log_dir="/tmp/logs")
    assert init._log_dir == "/tmp/logs"


def test_inspect_initializer_register_all_targets_default() -> None:
    from pyrit.inspect_bridge import InspectInitializer

    init = InspectInitializer()
    assert init._register_all_targets is True


def test_inspect_initializer_register_all_targets_false() -> None:
    from pyrit.inspect_bridge import InspectInitializer

    init = InspectInitializer(register_all_targets=False)
    assert init._register_all_targets is False


def test_required_env_vars_returns_empty_list() -> None:
    from pyrit.inspect_bridge import InspectInitializer

    init = InspectInitializer()
    assert init.required_env_vars == []


async def test_initialize_async_runs_without_error() -> None:
    from pyrit.inspect_bridge import InspectInitializer

    with patch("pyrit.inspect_bridge._initializer.require_inspect_ai"):
        init = InspectInitializer()
        await init.initialize_async()


async def test_initialize_async_is_idempotent() -> None:
    from pyrit.inspect_bridge import InspectInitializer

    with patch("pyrit.inspect_bridge._initializer.require_inspect_ai"):
        init = InspectInitializer()
        await init.initialize_async()
        await init.initialize_async()


async def test_initialize_async_raises_if_inspect_missing() -> None:
    from pyrit.inspect_bridge import InspectBridgeError, InspectInitializer

    with patch(
        "pyrit.inspect_bridge._initializer.require_inspect_ai",
        side_effect=InspectBridgeError(message="inspect_ai not installed"),
    ):
        init = InspectInitializer()
        try:
            await init.initialize_async()
            raise AssertionError("Should have raised InspectBridgeError")
        except InspectBridgeError:
            pass


async def test_initialize_async_registers_targets() -> None:
    from pyrit.inspect_bridge import InspectInitializer

    mock_target = MagicMock()
    mock_target.get_identifier.return_value.unique_name = "my_test_target"

    mock_entry = MagicMock()
    mock_entry.instance = mock_target

    mock_registry = MagicMock()
    mock_registry.get_all_instances.return_value = [mock_entry]

    with (
        patch("pyrit.inspect_bridge._initializer.require_inspect_ai"),
        patch("pyrit.inspect_bridge._initializer.TargetRegistry") as mock_tr_class,
    ):
        mock_tr_class.get_registry_singleton.return_value = mock_registry
        init = InspectInitializer(register_all_targets=True)
        await init.initialize_async()

    mock_registry.get_all_instances.assert_called_once()
