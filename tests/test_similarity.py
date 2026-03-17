"""Tests for the similarity scoring module."""

import pytest

from src.similarity import SimilarityScorer


@pytest.fixture(scope="module")
def scorer() -> SimilarityScorer:
    """Shared scorer instance (model loaded once per test module)."""
    s = SimilarityScorer(
        model_name="all-MiniLM-L6-v2",
        fuzzy_weight=0.4,
        semantic_weight=0.6,
    )
    s.load_model()
    return s


class TestSimilarityScorerInit:
    def test_invalid_weights(self) -> None:
        with pytest.raises(ValueError, match="must equal 1.0"):
            SimilarityScorer(fuzzy_weight=0.5, semantic_weight=0.6)

    def test_valid_weights(self) -> None:
        s = SimilarityScorer(fuzzy_weight=0.3, semantic_weight=0.7)
        assert s.fuzzy_weight == 0.3
        assert s.semantic_weight == 0.7


class TestFuzzyScore:
    def test_exact_match(self, scorer: SimilarityScorer) -> None:
        assert scorer.fuzzy_score("hello world", "hello world") == 1.0

    def test_empty_string(self, scorer: SimilarityScorer) -> None:
        assert scorer.fuzzy_score("", "hello") == 0.0
        assert scorer.fuzzy_score("hello", "") == 0.0

    def test_partial_match(self, scorer: SimilarityScorer) -> None:
        score = scorer.fuzzy_score("atlas logistics", "atlas logistic")
        assert 0.8 < score <= 1.0

    def test_no_match(self, scorer: SimilarityScorer) -> None:
        score = scorer.fuzzy_score("xyz123", "abcdef")
        assert score < 0.3


class TestSemanticScore:
    def test_identical_strings(self, scorer: SimilarityScorer) -> None:
        score = scorer.semantic_score("Azerbaijan Railways", "Azerbaijan Railways")
        assert score > 0.99

    def test_similar_meaning(self, scorer: SimilarityScorer) -> None:
        score = scorer.semantic_score("railway company", "train transport")
        assert score > 0.3

    def test_empty_string(self, scorer: SimilarityScorer) -> None:
        assert scorer.semantic_score("", "hello") == 0.0


class TestCombinedScore:
    def test_exact_match_high_score(self, scorer: SimilarityScorer) -> None:
        score = scorer.combined_score("Atlas Logistics", "Atlas Logistics")
        assert score > 0.95

    def test_custom_weights(self, scorer: SimilarityScorer) -> None:
        score = scorer.combined_score(
            "Atlas", "Atlas Logistics",
            fuzzy_weight=0.8, semantic_weight=0.2,
        )
        assert 0.0 <= score <= 1.0


class TestBatchScores:
    def test_batch_fuzzy(self, scorer: SimilarityScorer) -> None:
        scores = scorer.batch_fuzzy_scores("atlas", ["atlas", "atlaz", "xyz"])
        assert len(scores) == 3
        assert scores[0] > scores[2]

    def test_batch_semantic(self, scorer: SimilarityScorer) -> None:
        scores = scorer.batch_semantic_scores(
            "railway", ["railway company", "oil company", "xyz"]
        )
        assert len(scores) == 3
        assert scores[0] > scores[2]

    def test_batch_empty_candidates(self, scorer: SimilarityScorer) -> None:
        scores = scorer.batch_semantic_scores("hello", [])
        assert scores == []
