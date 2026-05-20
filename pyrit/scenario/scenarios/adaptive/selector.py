# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Epsilon-greedy selector and context extractors for adaptive scenarios."""

from __future__ import annotations

import random
import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrit.models.seeds.seed_attack_group import SeedAttackGroup

"""Maps a ``SeedAttackGroup`` to an adaptive context key."""
ContextExtractor = Callable[["SeedAttackGroup"], str]
"""Default context: all objectives share one selection table."""
GLOBAL_CONTEXT: str = "_global"
"""Fallback context for seed groups with no harm category metadata."""
UNCATEGORIZED_CONTEXT: str = "_uncategorized"


def global_context(_seed_attack_group: SeedAttackGroup) -> str:
    """
    Return a single shared context for all objectives.

    Returns:
        str: Always :data:`GLOBAL_CONTEXT`.
    """
    return GLOBAL_CONTEXT


def harm_category_context(seed_attack_group: SeedAttackGroup) -> str:
    """
    Return a context keyed by the sorted, ``|``-joined harm categories.

    Multi-category seeds form their own bucket; sorting makes the key deterministic.

    Returns:
        str: The ``|``-joined sorted harm categories, or :data:`UNCATEGORIZED_CONTEXT`
            when the seed group has no categories.
    """
    categories = seed_attack_group.harm_categories
    if not categories:
        return UNCATEGORIZED_CONTEXT
    return "|".join(sorted(categories))


class AdaptiveTechniqueSelector:
    """
    Epsilon-greedy selector over attack techniques.

    Maintains a ``(context, technique) -> (successes, attempts)`` table. With
    probability ``epsilon`` picks uniformly at random; otherwise picks the
    technique with the highest Laplace-smoothed estimate ``(s + 1) / (n + 1)``
    (unseen techniques start at 1.0). A ``(context, technique)`` cell with
    fewer than ``pool_threshold`` attempts falls back to the technique's
    pooled rate across all contexts.

    All public methods are guarded by a ``threading.Lock`` so concurrent
    callers cannot corrupt the table. The lock makes individual ops atomic,
    not the overall select → execute → record sequence.
    """

    # Tolerance for tiebreaking on float estimates (current estimates are exact
    # rationals; this guards against future estimator changes).
    _TIE_TOL: float = 1e-12

    def __init__(
        self,
        *,
        epsilon: float = 0.2,
        pool_threshold: int = 3,
        rng: random.Random | None = None,
    ) -> None:
        """
        Args:
            epsilon (float): Exploration probability in [0.0, 1.0]. Defaults to 0.2.
            pool_threshold (int): Minimum per-(context, technique) attempts before
                the local estimate replaces the pooled rate. Must be >= 1; set to 1
                to disable pooling. Defaults to 3.
            rng (random.Random | None): RNG for reproducible decisions. Defaults
                to a fresh unseeded ``random.Random()``.

        Raises:
            ValueError: If ``epsilon`` is outside [0.0, 1.0] or ``pool_threshold`` < 1.
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0.0, 1.0], got {epsilon}")
        if pool_threshold < 1:
            raise ValueError(f"pool_threshold must be >= 1, got {pool_threshold}")

        self._epsilon = epsilon
        self._pool_threshold = pool_threshold
        self._rng = rng if rng is not None else random.Random()
        self._counts: dict[tuple[str, str], tuple[int, int]] = {}
        # Per-technique pooled counts, kept in sync with ``_counts`` so the
        # pooled-backoff branch in ``_estimate`` is O(1).
        self._global_counts: dict[str, tuple[int, int]] = {}
        # Guards _counts, _global_counts, and _rng against concurrent callers.
        self._lock = threading.Lock()

    def select(self, *, context: str, techniques: Sequence[str]) -> str:
        """
        Pick the next technique to try for ``context``.

        Args:
            context (str): The context key.
            techniques (Sequence[str]): Candidate technique names.

        Returns:
            str: The chosen technique name.

        Raises:
            ValueError: If ``techniques`` is empty.
        """
        technique_list = list(techniques)
        if not technique_list:
            raise ValueError("techniques must contain at least one entry")

        with self._lock:
            if self._rng.random() < self._epsilon:
                return self._rng.choice(technique_list)

            estimates = {t: self._estimate(context=context, technique=t) for t in technique_list}
            best = max(estimates.values())
            winners = [t for t, value in estimates.items() if value >= best - self._TIE_TOL]
            return self._rng.choice(winners)

    def record_outcome(self, *, context: str, technique: str, success: bool) -> None:
        """
        Record the outcome of an attempt.

        Args:
            context (str): The context key the decision was made under.
            technique (str): The technique that was tried.
            success (bool): Whether the attempt succeeded.
        """
        with self._lock:
            successes, attempts = self._counts.get((context, technique), (0, 0))
            attempts += 1
            if success:
                successes += 1
            self._counts[(context, technique)] = (successes, attempts)

            global_successes, global_attempts = self._global_counts.get(technique, (0, 0))
            global_attempts += 1
            if success:
                global_successes += 1
            self._global_counts[technique] = (global_successes, global_attempts)

    def success_rate(self, *, context: str, technique: str) -> float:
        """Return the Laplace-smoothed estimate ``(s + 1) / (n + 1)`` used for exploitation."""
        with self._lock:
            return self._estimate(context=context, technique=technique)

    def counts(self, *, context: str, technique: str) -> tuple[int, int]:
        """Return raw ``(successes, attempts)`` for a ``(context, technique)`` cell."""
        with self._lock:
            return self._counts.get((context, technique), (0, 0))

    def snapshot(self) -> dict[tuple[str, str], tuple[int, int]]:
        """Return a shallow copy of the full counts table (for logging/debug)."""
        with self._lock:
            return dict(self._counts)

    def _estimate(self, *, context: str, technique: str) -> float:
        """
        Estimate for ``(context, technique)``; falls back to pooled rate below
        ``pool_threshold`` local attempts.

        Callers must already hold ``self._lock``.

        Returns:
            float: Laplace-smoothed success-rate estimate in ``(0, 1)``.
        """
        local_s, local_n = self._counts.get((context, technique), (0, 0))
        if local_n >= self._pool_threshold:
            return (local_s + 1) / (local_n + 1)
        global_s, global_n = self._global_counts.get(technique, (0, 0))
        return (global_s + 1) / (global_n + 1)
