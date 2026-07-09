# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Best-effort compatibility shims for plug-ins built against a slightly different PyRIT.

A plug-in wheel is authored against one PyRIT version, then loaded into whatever PyRIT the
operator is running. When the host API has drifted, small mechanical differences — most
commonly a renamed extension-point method — can leave the plug-in's scenarios abstract and
therefore undiscoverable, even though the plug-in's own logic is unchanged. Rather than
force every plug-in author to re-release on each host bump, the loader applies narrow, loud
heuristics that bridge known mechanical renames so minor drift does not block loading.

The shims are deliberately conservative. A scenario is bridged only when it is abstract
*solely* because of a recognized rename and a usable predecessor method is present; a class
abstract for any other reason is left alone (and fails loudly downstream). Every bridge and
every detected version mismatch logs a loud, greppable warning so the drift is visible and
the operator knows that rebuilding the plug-in against the running PyRIT removes the shim.
This module owns drift-bridging so the loader stays focused on extract/import/register.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from pathlib import Path
    from typing import Any

logger = logging.getLogger(__name__)

# Renamed scenario extension points: each maps a current (host) abstract method to the
# older predecessor a plug-in may still define. A predecessor took no ``context`` (it read
# ``self._*`` state the base still populates before the build call), so the bridge can
# delegate to it and ignore ``context``. Add an entry here when a future rename needs the
# same mechanical bridge.
_SCENARIO_METHOD_RENAMES: dict[str, str] = {
    "_build_atomic_attacks_async": "_get_atomic_attacks_async",
}


def warn_on_version_drift(*, extract_dir: Path) -> str | None:
    """
    Log a loud warning when a plug-in was built against a different PyRIT minor version.

    Reads the plug-in's ``Requires-Dist: pyrit`` pin from its wheel ``METADATA`` and
    compares it to the running ``pyrit`` version. A mismatch is surfaced but never fatal:
    the loader tolerates slight drift and lets the import/registration path (plus the
    scenario shims below) decide whether the plug-in actually works. Absence of a pin, or
    an unparseable one, is silently ignored — this is a best-effort signal, not a gate.

    Args:
        extract_dir: The directory the plug-in wheel was extracted to.

    Returns:
        str | None: The plug-in's declared PyRIT pin when a drift warning was emitted,
        else ``None``.
    """
    declared = _read_required_pyrit_version(extract_dir=extract_dir)
    if declared is None:
        return None

    import pyrit

    running = getattr(pyrit, "__version__", "") or ""
    if _same_minor(declared, running):
        return None

    logger.warning(
        "PLUGIN VERSION DRIFT: plug-in was built against pyrit %s but pyrit %s is running. "
        "Proceeding; compatibility shims will bridge known mechanical differences, but "
        "rebuild the plug-in against the running pyrit if scenarios fail to load.",
        declared,
        running or "(unknown)",
    )
    return declared


def bridge_scenario_extension_points(*, package_name: str) -> list[str]:
    """
    Make a plug-in's scenarios concrete by bridging renamed extension-point methods.

    Enumerates concrete-intent ``Scenario`` subclasses owned by ``package_name`` that are
    still abstract, and for each one whose only unimplemented abstract methods are known
    renames with a usable predecessor, injects a thin adapter so the class satisfies the
    current contract. Classes abstract for any other reason are left untouched.

    Args:
        package_name: The plug-in's top-level package name.

    Returns:
        list[str]: Human-readable descriptions of the bridges applied, for logging.
    """
    from pyrit.scenario.core import Scenario

    prefix = f"{package_name}."
    applied: list[str] = []

    for cls in _iter_owned_subclasses(base=Scenario, package_name=package_name, prefix=prefix):
        if not inspect.isabstract(cls):
            continue

        missing = set(cls.__abstractmethods__)  # type: ignore[ty:unresolved-attribute]
        # Only bridge when every missing method is a known rename we can satisfy from a
        # predecessor the class actually provides. Otherwise the class is abstract for a
        # reason we must not paper over, so leave it (it will fail loudly downstream).
        if not missing or not missing.issubset(_SCENARIO_METHOD_RENAMES.keys()):
            continue
        if not all(_has_concrete_method(cls, _SCENARIO_METHOD_RENAMES[name]) for name in missing):
            continue

        for new_name in missing:
            predecessor = _SCENARIO_METHOD_RENAMES[new_name]
            setattr(cls, new_name, _make_rename_bridge(predecessor_name=predecessor))
            applied.append(f"{cls.__name__}.{predecessor} -> {new_name}")
            logger.warning(
                "PLUGIN COMPAT SHIM: bridged %s.%s to the renamed %s (plug-in built against "
                "an older pyrit). Rebuild the plug-in against the running pyrit to remove this shim.",
                cls.__name__,
                predecessor,
                new_name,
            )
        cls.__abstractmethods__ = frozenset(cls.__abstractmethods__ - missing)  # type: ignore[ty:unresolved-attribute]

    return applied


def _make_rename_bridge(*, predecessor_name: str) -> Callable[..., Coroutine[Any, Any, Any]]:
    """
    Build an adapter that satisfies a renamed async extension point via its predecessor.

    Args:
        predecessor_name: The older method name the plug-in still defines.

    Returns:
        Callable[..., Coroutine[Any, Any, Any]]: An async adapter that ignores the new
        ``context`` keyword and delegates to the predecessor (which reads ``self._*``
        state the base populates before the build call).
    """

    async def _bridged_build_async(self, **_kwargs: Any) -> Any:  # noqa: ANN001
        return await getattr(self, predecessor_name)()

    _bridged_build_async.__name__ = predecessor_name
    _bridged_build_async.__qualname__ = predecessor_name
    return _bridged_build_async


def _has_concrete_method(cls: type, name: str) -> bool:
    """
    Return whether ``cls`` provides a concrete (non-abstract) method of the given name.

    Args:
        cls: The class to inspect.
        name: The method name to look for.

    Returns:
        bool: True when the method exists and is not itself abstract.
    """
    method = getattr(cls, name, None)
    return callable(method) and name not in getattr(cls, "__abstractmethods__", frozenset())


def _iter_owned_subclasses(*, base: type, package_name: str, prefix: str) -> list[type]:
    """
    Return every subclass of ``base`` whose module is owned by the plug-in package.

    Args:
        base: The base class to enumerate subclasses of.
        package_name: The plug-in's top-level package name.
        prefix: ``f"{package_name}."`` (passed in to avoid recomputation).

    Returns:
        list[type]: The owned subclasses currently loaded in memory.
    """
    seen: set[int] = set()
    owned: list[type] = []
    stack = list(base.__subclasses__())
    while stack:
        cls = stack.pop()
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        stack.extend(cls.__subclasses__())
        module = cls.__module__ or ""
        if module == package_name or module.startswith(prefix):
            owned.append(cls)
    return owned


def _read_required_pyrit_version(*, extract_dir: Path) -> str | None:
    """
    Return the pinned ``pyrit`` version from the plug-in's wheel ``METADATA``, if any.

    Args:
        extract_dir: The directory the plug-in wheel was extracted to.

    Returns:
        str | None: The pinned version string (e.g. ``"0.14.0"``) or ``None`` when no
        parseable ``Requires-Dist: pyrit`` pin is present.
    """
    import re

    for metadata in extract_dir.glob("*.dist-info/METADATA"):
        try:
            text = metadata.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if not line.lower().startswith("requires-dist:"):
                continue
            if not re.match(r"requires-dist:\s*pyrit\b", line, flags=re.IGNORECASE):
                continue
            match = re.search(r"==\s*([0-9][0-9A-Za-z.\-]*)", line)
            if match:
                return match.group(1)
    return None


def _same_minor(left: str, right: str) -> bool:
    """
    Return whether two version strings share the same ``major.minor`` prefix.

    Args:
        left: A version string (e.g. ``"0.14.0"``).
        right: A version string (e.g. ``"0.15.0.dev0"``).

    Returns:
        bool: True when both parse to the same ``(major, minor)``; False otherwise. An
        unparseable side compares unequal so drift is surfaced rather than hidden.
    """
    return _major_minor(left) == _major_minor(right) and _major_minor(left) is not None


def _major_minor(version: str) -> tuple[int, int] | None:
    """
    Parse the leading ``major.minor`` from a version string.

    Args:
        version: A version string.

    Returns:
        tuple[int, int] | None: The ``(major, minor)`` pair, or ``None`` when the string
        does not begin with two integer components.
    """
    parts = version.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None
