# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Scorer decorators that wrap a ``Scorer`` to add behavior without subclassing."""

from pyrit.score.decorators.outcome_scorer import OutcomeScorer

__all__ = ["OutcomeScorer"]
