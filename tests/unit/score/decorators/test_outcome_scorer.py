# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for ``pyrit.score.decorators.outcome_scorer``."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from pyrit.identifiers import ComponentIdentifier
from pyrit.models import Message, MessagePiece, Score
from pyrit.score import Scorer
from pyrit.score.decorators import OutcomeScorer


def _make_score(*, value: str, score_type: str = "true_false") -> Score:
    return Score(
        score_value=value,
        score_value_description="",
        score_type=score_type,  # type: ignore[arg-type]
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
