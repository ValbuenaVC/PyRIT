# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``OutcomeScorer`` — wrap a ``Scorer`` to resolve its output into a transition label.

This is the bridge between scorer outputs and state-machine transitions for
the scenario core refactor. It is a composition wrapper, not a ``Scorer``
subclass: callers that need the full ``Scorer`` interface use
``OutcomeScorer.wrapped_scorer`` directly.

Part of the scenario core refactor scaffold (Phase 1). ``ScenarioStep``
implementations consume ``OutcomeScorer.resolve_outcome_async`` to map a
freshly produced response into one of the step's declared ``outputs``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyrit.models import Message, Score
    from pyrit.score import Scorer


class OutcomeScorer:
    """
    Decorator that maps a wrapped ``Scorer``'s output to a transition label.

    The ``outcome_map`` is an ordered ``dict`` of ``label -> predicate``. The
    first label whose predicate returns ``True`` for any produced ``Score`` is
    emitted. If no predicate matches (or the wrapped scorer returns no
    scores), the sentinel ``"unscored"`` is emitted; the surrounding policy
    can declare an explicit ``"unscored" -> <state>`` transition if needed.
    """

    #: Sentinel label emitted when no entry in ``outcome_map`` matches and
    #: when the wrapped scorer returns no scores at all.
    UNSCORED: ClassVar[str] = "unscored"

    def __init__(
        self,
        *,
        wrapped_scorer: Scorer,
        outcome_map: dict[str, Callable[[Score], bool]],
    ) -> None:
        """
        Initialize an ``OutcomeScorer``.

        Args:
            wrapped_scorer (Scorer): The scorer whose output drives outcome
                resolution. The wrapper does not modify the scorer; it only
                calls ``score_async`` and maps the result.
            outcome_map (dict[str, Callable[[Score], bool]]): Ordered mapping
                from transition label to a predicate over ``Score``. The first
                label whose predicate matches any produced score is emitted.
                Reserved label ``"unscored"`` must not appear here.

        Raises:
            ValueError: If ``outcome_map`` is empty or contains the reserved
                ``"unscored"`` label.
        """
        if not outcome_map:
            raise ValueError("OutcomeScorer requires a non-empty outcome_map.")
        if self.UNSCORED in outcome_map:
            raise ValueError(
                f"Label {self.UNSCORED!r} is reserved as the no-match sentinel "
                f"and may not appear in outcome_map."
            )

        self._wrapped_scorer = wrapped_scorer
        self._outcome_map = dict(outcome_map)

    @property
    def wrapped_scorer(self) -> Scorer:
        """Return the wrapped scorer for callers needing the full Scorer interface."""
        return self._wrapped_scorer

    @property
    def outcomes(self) -> list[str]:
        """
        Return the full list of labels this scorer can emit.

        Includes the implicit ``"unscored"`` sentinel at the end, so step
        validation (``set(outcomes).issubset(step.outputs)``) catches missing
        ``"unscored"`` declarations early.

        Returns:
            list[str]: Ordered labels, with ``"unscored"`` last.
        """
        return [*self._outcome_map.keys(), self.UNSCORED]

    async def resolve_outcome_async(
        self,
        message: Message,
        *,
        objective: str | None = None,
    ) -> str:
        """
        Score ``message`` with the wrapped scorer and return the first matching label.

        Args:
            message (Message): The message to score.
            objective (str | None): Optional objective forwarded to the
                wrapped scorer's ``score_async``.

        Returns:
            str: The first matching label from ``outcome_map``, or
                ``"unscored"`` if no entry matches or the wrapped scorer
                returned no scores.
        """
        scores = await self._wrapped_scorer.score_async(message, objective=objective)
        if not scores:
            return self.UNSCORED

        for label, predicate in self._outcome_map.items():
            if any(predicate(score) for score in scores):
                return label

        return self.UNSCORED
