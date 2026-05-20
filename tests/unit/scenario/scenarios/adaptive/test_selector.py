# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from unittest.mock import MagicMock

import pytest

from pyrit.scenario.scenarios.adaptive.selector import (
    GLOBAL_CONTEXT,
    UNCATEGORIZED_CONTEXT,
    AdaptiveTechniqueSelector,
    global_context,
    harm_category_context,
)

TECHNIQUES = ["a", "b", "c", "d"]


def _seeded_selector(*, epsilon: float = 0.0, pool_threshold: int = 3, seed: int = 0) -> AdaptiveTechniqueSelector:
    return AdaptiveTechniqueSelector(
        epsilon=epsilon,
        pool_threshold=pool_threshold,
        rng=random.Random(seed),
    )


class TestAdaptiveTechniqueSelectorInit:
    def test_init_defaults(self):
        selector = AdaptiveTechniqueSelector()
        assert selector.snapshot() == {}

    @pytest.mark.parametrize("bad_epsilon", [-0.1, 1.1, 2.0, -1.0])
    def test_init_rejects_out_of_range_epsilon(self, bad_epsilon):
        with pytest.raises(ValueError, match="epsilon"):
            AdaptiveTechniqueSelector(epsilon=bad_epsilon)

    def test_init_rejects_pool_threshold_below_one(self):
        with pytest.raises(ValueError, match="pool_threshold"):
            AdaptiveTechniqueSelector(pool_threshold=0)
        with pytest.raises(ValueError, match="pool_threshold"):
            AdaptiveTechniqueSelector(pool_threshold=-1)


class TestAdaptiveTechniqueSelectorSelect:
    def test_select_empty_techniques_raises(self):
        selector = _seeded_selector()
        with pytest.raises(ValueError, match="techniques"):
            selector.select(context=GLOBAL_CONTEXT, techniques=[])

    def test_select_all_unseen_ties_resolved_randomly(self):
        # With epsilon=0 and an empty table, every technique has estimate 1/1=1.0,
        # so the result is the seeded random tiebreak. Different seeds should
        # be able to produce different winners.
        winners = {_seeded_selector(seed=s).select(context=GLOBAL_CONTEXT, techniques=TECHNIQUES) for s in range(50)}
        assert len(winners) > 1
        assert winners.issubset(set(TECHNIQUES))

    def test_select_exploits_clear_winner(self):
        selector = _seeded_selector(pool_threshold=1)
        # Give "b" a track record of pure success, others pure failure.
        for _ in range(5):
            selector.record_outcome(context=GLOBAL_CONTEXT, technique="b", success=True)
        for technique in ("a", "c", "d"):
            for _ in range(5):
                selector.record_outcome(context=GLOBAL_CONTEXT, technique=technique, success=False)

        # With epsilon=0, every selection must exploit the winner.
        for _ in range(20):
            assert selector.select(context=GLOBAL_CONTEXT, techniques=TECHNIQUES) == "b"

    def test_select_epsilon_one_is_pure_random(self):
        selector = _seeded_selector(epsilon=1.0)
        # Bias the table heavily toward "a"; with epsilon=1 it must still be ignored.
        for _ in range(20):
            selector.record_outcome(context=GLOBAL_CONTEXT, technique="a", success=True)

        picks = [selector.select(context=GLOBAL_CONTEXT, techniques=TECHNIQUES) for _ in range(200)]
        assert set(picks) == set(TECHNIQUES)

    def test_select_epsilon_zero_never_explores(self):
        selector = _seeded_selector(epsilon=0.0, pool_threshold=1)
        for _ in range(3):
            selector.record_outcome(context=GLOBAL_CONTEXT, technique="a", success=True)
        # Make the other techniques tried-and-failed so they fall below "a"'s estimate;
        # unseen techniques would otherwise tie at the optimistic 1.0.
        for technique in ("b", "c", "d"):
            selector.record_outcome(context=GLOBAL_CONTEXT, technique=technique, success=False)
        for _ in range(50):
            assert selector.select(context=GLOBAL_CONTEXT, techniques=TECHNIQUES) == "a"

    def test_select_cold_start_round_robins(self):
        # Optimistic init + epsilon=0: untried techniques tie at 1.0 and beat tried-and-failed
        # techniques (1/2 = 0.5). So the first failures push each technique to "tried" exactly once
        # before any technique gets tried twice.
        selector = _seeded_selector(pool_threshold=1)
        tried: list[str] = []
        for _ in range(len(TECHNIQUES)):
            technique = selector.select(context=GLOBAL_CONTEXT, techniques=TECHNIQUES)
            tried.append(technique)
            selector.record_outcome(context=GLOBAL_CONTEXT, technique=technique, success=False)
        assert sorted(tried) == sorted(TECHNIQUES)


class TestAdaptiveTechniqueSelectorUpdate:
    def test_record_outcome_accumulates_counts(self):
        selector = _seeded_selector()
        selector.record_outcome(context="ctx", technique="a", success=True)
        selector.record_outcome(context="ctx", technique="a", success=False)
        selector.record_outcome(context="ctx", technique="a", success=True)
        assert selector.counts(context="ctx", technique="a") == (2, 3)

    def test_record_outcome_separate_contexts_are_independent(self):
        selector = _seeded_selector()
        selector.record_outcome(context="x", technique="a", success=True)
        selector.record_outcome(context="y", technique="a", success=False)
        assert selector.counts(context="x", technique="a") == (1, 1)
        assert selector.counts(context="y", technique="a") == (0, 1)

    def test_counts_default_zero_for_unseen(self):
        selector = _seeded_selector()
        assert selector.counts(context="missing", technique="missing") == (0, 0)

    def test_record_outcome_keeps_pooled_global_counts_in_sync(self):
        # Pooled-global counts back the O(1) pooled-backoff branch in _estimate.
        # They must aggregate across contexts for a given technique.
        selector = _seeded_selector(pool_threshold=5)
        selector.record_outcome(context="x", technique="a", success=True)
        selector.record_outcome(context="y", technique="a", success=False)
        selector.record_outcome(context="z", technique="a", success=True)
        selector.record_outcome(context="x", technique="b", success=True)

        # Below the local threshold, _estimate must use the pooled-global rate.
        # technique "a": 2 successes / 3 attempts -> (2+1)/(3+1) = 0.75
        assert selector.success_rate(context="new_ctx", technique="a") == pytest.approx(0.75)
        # technique "b": 1/1 -> (1+1)/(1+1) = 1.0
        assert selector.success_rate(context="new_ctx", technique="b") == pytest.approx(1.0)
        # Unseen technique "c" -> (0+1)/(0+1) = 1.0
        assert selector.success_rate(context="new_ctx", technique="c") == pytest.approx(1.0)


class TestAdaptiveTechniqueSelectorEstimate:
    def test_success_rate_unseen_is_one(self):
        # Optimistic init: (0 + 1) / (0 + 1) = 1.0
        selector = _seeded_selector()
        assert selector.success_rate(context="ctx", technique="a") == pytest.approx(1.0)

    def test_success_rate_local_when_above_threshold(self):
        selector = _seeded_selector(pool_threshold=2)
        for _ in range(3):
            selector.record_outcome(context="ctx", technique="a", success=True)
        # (3 + 1) / (3 + 1) = 1.0
        assert selector.success_rate(context="ctx", technique="a") == pytest.approx(1.0)

    def test_success_rate_pools_when_below_threshold(self):
        selector = _seeded_selector(pool_threshold=5)
        # Local cell has only 1 attempt (below threshold).
        selector.record_outcome(context="ctx", technique="a", success=False)
        # Other contexts have plenty of data for technique "a".
        for _ in range(10):
            selector.record_outcome(context="other", technique="a", success=True)
        # Pooled estimate = (10 + 0 + 1) / (10 + 1 + 1) = 11/12.
        assert selector.success_rate(context="ctx", technique="a") == pytest.approx(11 / 12)


class TestContextExtractors:
    def test_global_context_is_constant(self):
        sg = MagicMock()
        assert global_context(sg) == GLOBAL_CONTEXT

    def test_harm_category_context_joins_sorted_categories(self):
        sg = MagicMock()
        sg.harm_categories = ["violence", "hate"]
        # Multi-category seeds form their own bucket; sorting keeps the key deterministic.
        assert harm_category_context(sg) == "hate|violence"

    def test_harm_category_context_single_category(self):
        sg = MagicMock()
        sg.harm_categories = ["violence"]
        assert harm_category_context(sg) == "violence"

    def test_harm_category_context_falls_back_when_empty(self):
        sg = MagicMock()
        sg.harm_categories = []
        assert harm_category_context(sg) == UNCATEGORIZED_CONTEXT

    def test_harm_category_context_falls_back_when_none(self):
        sg = MagicMock()
        sg.harm_categories = None
        assert harm_category_context(sg) == UNCATEGORIZED_CONTEXT


class TestAdaptiveTechniqueSelectorConcurrency:
    """Concurrent record_outcome / select calls must not corrupt counts."""

    def test_concurrent_record_outcome_preserves_total_attempts(self):
        import threading

        selector = _seeded_selector(pool_threshold=1)
        threads_per_arm = 8
        attempts_per_thread = 100
        techniques = ["a", "b", "c", "d"]

        def worker(technique: str, success_pattern: list[bool]) -> None:
            for ok in success_pattern:
                selector.record_outcome(context=GLOBAL_CONTEXT, technique=technique, success=ok)

        threads: list[threading.Thread] = []
        expected_successes: dict[str, int] = dict.fromkeys(techniques, 0)
        for t in techniques:
            for i in range(threads_per_arm):
                pattern = [(j + i) % 2 == 0 for j in range(attempts_per_thread)]
                expected_successes[t] += sum(pattern)
                threads.append(threading.Thread(target=worker, args=(t, pattern)))

        for th in threads:
            th.start()
        for th in threads:
            th.join()

        # Every increment landed: no lost updates from interleaved read-modify-write.
        for t in techniques:
            successes, attempts = selector.counts(context=GLOBAL_CONTEXT, technique=t)
            assert attempts == threads_per_arm * attempts_per_thread
            assert successes == expected_successes[t]
