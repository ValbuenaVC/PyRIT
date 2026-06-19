# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Registry shims for the Inspect AI model provider and hooks.

This module is referenced by the ``[project.entry-points.inspect_ai]`` entry
point in ``pyproject.toml`` so that Inspect AI discovers the PyRIT model
provider and hooks automatically. Heavy imports of ``inspect_ai`` are deferred
to the decorated functions/classes so that importing this module does not
trigger an ``import inspect_ai``.
"""
