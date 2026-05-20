# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for ``pyrit.score.decorators.outcome_scorer``."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from pyrit.identifiers import ComponentIdentifier
from pyrit.models import Message, MessagePiece, Score
from pyrit.score import Scorer
from pyrit.score.decorators import OutcomeScorer

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_score(*, value: str, score_type: str = "true_false") -> Score:
    return Score(
        score_value=value,
        score_value_description="",
        score_type=score_type,  # type: ignore[ty:invalid-argument-type]
        score_rationale="",
        message_piece_id=str(uuid4()),
        scorer_class_identifier=ComponentIdentifier(
            class_name="MockScorer",
            class_module="tests.unit.score.decorators.test_outcome_scorer",
        ),
    )


def _make_message(*, text: str = "hello") -> Message:
    return Message(message_pieces=[MessagePiece(role="user", original_value=text)])


def test_init_rejects_empty_outcome_map():
    scorer = MagicMock(spec=Scorer)
    with pytest.raises(ValueError, match="non-empty outcome_map"):
        OutcomeScorer(wrapped_scorer=scorer, outcome_map={})


def test_init_rejects_reserved_unscored_label():
    scorer = MagicMock(spec=Scorer)
    with pytest.raises(ValueError, match="reserved"):
        OutcomeScorer(
            wrapped_scorer=scorer,
            outcome_map={"unscored": lambda s: True},
        )


def test_wrapped_scorer_is_exposed():
    inner = MagicMock(spec=Scorer)
    outer = OutcomeScorer(
        wrapped_scorer=inner,
        outcome_map={"hit": lambda s: True},
    )
    assert outer.wrapped_scorer is inner


def test_outcomes_includes_unscored_last():
    scorer = MagicMock(spec=Scorer)
    outer = OutcomeScorer(
        wrapped_scorer=scorer,
        outcome_map={
            "violation": lambda s: s.score_value == "true",
            "refusal": lambda s: s.score_value == "false",
        },
    )
    assert outer.outcomes == ["violation", "refusal", "unscored"]


async def test_resolve_outcome_returns_first_matching_label():
    scorer = MagicMock(spec=Scorer)
    scorer.score_async = AsyncMock(return_value=[_make_score(value="true")])
    outer = OutcomeScorer(
        wrapped_scorer=scorer,
        outcome_map={
            "violation": lambda s: s.score_value == "true",
            "refusal": lambda s: s.score_value == "false",
        },
    )

    label = await outer.resolve_outcome_async(_make_message())
    assert label == "violation"


async def test_resolve_outcome_preserves_outcome_map_order():
    """When multiple predicates match, the first label wins."""
    scorer = MagicMock(spec=Scorer)
    scorer.score_async = AsyncMock(return_value=[_make_score(value="true")])
    outer = OutcomeScorer(
        wrapped_scorer=scorer,
        outcome_map={
            "first": lambda s: True,
            "second": lambda s: True,
        },
    )

    label = await outer.resolve_outcome_async(_make_message())
    assert label == "first"


async def test_resolve_outcome_returns_unscored_when_no_predicate_matches():
    scorer = MagicMock(spec=Scorer)
    scorer.score_async = AsyncMock(return_value=[_make_score(value="false")])
    outer = OutcomeScorer(
        wrapped_scorer=scorer,
        outcome_map={
            "violation": lambda s: s.score_value == "true",
        },
    )

    label = await outer.resolve_outcome_async(_make_message())
    assert label == "unscored"


async def test_resolve_outcome_returns_unscored_when_no_scores_produced():
    scorer = MagicMock(spec=Scorer)
    scorer.score_async = AsyncMock(return_value=[])
    outer = OutcomeScorer(
        wrapped_scorer=scorer,
        outcome_map={"hit": lambda s: True},
    )

    label = await outer.resolve_outcome_async(_make_message())
    assert label == "unscored"


async def test_resolve_outcome_forwards_objective_to_wrapped_scorer():
    scorer = MagicMock(spec=Scorer)
    scorer.score_async = AsyncMock(return_value=[_make_score(value="true")])
    outer = OutcomeScorer(
        wrapped_scorer=scorer,
        outcome_map={"hit": lambda s: True},
    )

    message = _make_message()
    await outer.resolve_outcome_async(message, objective="break the model")
    scorer.score_async.assert_called_once()
    assert scorer.score_async.call_args.kwargs["objective"] == "break the model"


async def test_resolve_outcome_matches_against_any_score_in_list():
    """Predicate firing on *any* score in the list is enough."""
    scorer = MagicMock(spec=Scorer)
    scorer.score_async = AsyncMock(
        return_value=[
            _make_score(value="false"),
            _make_score(value="false"),
            _make_score(value="true"),
        ]
    )
    outer = OutcomeScorer(
        wrapped_scorer=scorer,
        outcome_map={"hit": lambda s: s.score_value == "true"},
    )
    label = await outer.resolve_outcome_async(_make_message())
    assert label == "hit"


def test_unscored_sentinel_value_is_stable():
    """Lock the public sentinel string so downstream policies can declare it."""
    assert OutcomeScorer.UNSCORED == "unscored"


async def test_resolve_outcome_returns_unscored_when_wrapped_scorer_returns_none():
    """`if not scores` must treat ``None`` like an empty list, not crash."""
    scorer = MagicMock(spec=Scorer)
    scorer.score_async = AsyncMock(return_value=None)
    outer = OutcomeScorer(
        wrapped_scorer=scorer,
        outcome_map={"hit": lambda s: True},
    )

    label = await outer.resolve_outcome_async(_make_message())
    assert label == OutcomeScorer.UNSCORED
    assert label in outer.outcomes


async def test_resolve_outcome_unscored_is_declared_in_outcomes():
    """When no predicate matches, the returned sentinel must be in ``outcomes``."""
    scorer = MagicMock(spec=Scorer)
    scorer.score_async = AsyncMock(return_value=[_make_score(value="0.5", score_type="float_scale")])
    outer = OutcomeScorer(
        wrapped_scorer=scorer,
        outcome_map={
            "violation": lambda s: s.score_value == "true",
            "refusal": lambda s: s.score_value == "false",
        },
    )

    label = await outer.resolve_outcome_async(_make_message())
    assert label == OutcomeScorer.UNSCORED
    assert label in outer.outcomes


@pytest.mark.parametrize(
    ("scorer_values", "score_type"),
    [
        ([], "true_false"),
        ([None], "true_false"),
        (["true"], "true_false"),
        (["false"], "true_false"),
        (["0.5"], "float_scale"),
        (["false", "true"], "true_false"),
        (["true", "false"], "true_false"),
        (["0.0", "0.5", "1.0"], "float_scale"),
    ],
)
async def test_resolved_label_is_always_in_declared_outcomes(scorer_values, score_type):
    """Invariant: every label ``resolve_outcome_async`` returns is in ``outcomes``."""
    if scorer_values == [None]:
        return_value = None
    else:
        return_value = [_make_score(value=v, score_type=score_type) for v in scorer_values]
    scorer = MagicMock(spec=Scorer)
    scorer.score_async = AsyncMock(return_value=return_value)
    outer = OutcomeScorer(
        wrapped_scorer=scorer,
        outcome_map={
            "violation": lambda s: s.score_value == "true",
            "refusal": lambda s: s.score_value == "false",
        },
    )

    label = await outer.resolve_outcome_async(_make_message())
    assert label in outer.outcomes


async def test_resolve_outcome_propagates_wrapped_scorer_exception():
    """Errors from the wrapped scorer must bubble up; the wrapper is not a swallow."""
    scorer = MagicMock(spec=Scorer)
    scorer.score_async = AsyncMock(side_effect=RuntimeError("boom"))
    outer = OutcomeScorer(
        wrapped_scorer=scorer,
        outcome_map={"hit": lambda s: True},
    )

    with pytest.raises(RuntimeError, match="boom"):
        await outer.resolve_outcome_async(_make_message())


async def test_resolve_outcome_propagates_predicate_exception():
    """Errors from a predicate must bubble up rather than degrading to ``unscored``."""
    scorer = MagicMock(spec=Scorer)
    scorer.score_async = AsyncMock(return_value=[_make_score(value="true")])

    def _bad_predicate(score: Score) -> bool:
        raise ValueError("predicate failure")

    outer = OutcomeScorer(
        wrapped_scorer=scorer,
        outcome_map={"hit": _bad_predicate},
    )

    with pytest.raises(ValueError, match="predicate failure"):
        await outer.resolve_outcome_async(_make_message())


async def test_init_defensively_copies_outcome_map():
    """Mutating the input ``outcome_map`` after init must not affect resolution."""
    scorer = MagicMock(spec=Scorer)
    scorer.score_async = AsyncMock(return_value=[_make_score(value="true")])
    outcome_map: dict[str, Callable[[Score], bool]] = {
        "violation": lambda s: s.score_value == "true",
    }
    outer = OutcomeScorer(wrapped_scorer=scorer, outcome_map=outcome_map)

    outcome_map.clear()
    outcome_map["other"] = lambda s: True

    assert outer.outcomes == ["violation", OutcomeScorer.UNSCORED]
    label = await outer.resolve_outcome_async(_make_message())
    assert label == "violation"


def test_outcome_scorer_is_not_a_scorer_subclass():
    """``OutcomeScorer`` is a composition wrapper; the canonical identity path is
    ``wrapped_scorer.get_identifier()``. Locking this in prevents accidental
    subclassing that would shift identifier composition semantics."""
    inner = MagicMock(spec=Scorer)
    outer = OutcomeScorer(wrapped_scorer=inner, outcome_map={"hit": lambda s: True})

    assert not isinstance(outer, Scorer)
    assert not hasattr(outer, "get_identifier")
    assert outer.wrapped_scorer is inner
