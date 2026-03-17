"""Similarity scoring using fuzzy matching and semantic embeddings.

Provides token-based fuzzy scoring via rapidfuzz and semantic scoring
via sentence-BERT (all-MiniLM-L6-v2) cosine similarity.
"""

from typing import Optional

import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer


class SimilarityScorer:
    """Computes similarity between company names using fuzzy and semantic methods.

    Attributes:
        model_name: Sentence-BERT model identifier.
        fuzzy_weight: Weight for fuzzy score in combined calculation.
        semantic_weight: Weight for semantic score in combined calculation.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        fuzzy_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> None:
        """Initialize the scorer.

        Args:
            model_name: HuggingFace model name for sentence embeddings.
            fuzzy_weight: Weight for fuzzy score (0-1).
            semantic_weight: Weight for semantic score (0-1).
        """
        if abs((fuzzy_weight + semantic_weight) - 1.0) > 1e-6:
            raise ValueError("fuzzy_weight + semantic_weight must equal 1.0")

        self.model_name = model_name
        self.fuzzy_weight = fuzzy_weight
        self.semantic_weight = semantic_weight
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def load_model(self) -> None:
        """Explicitly load the model into memory."""
        _ = self.model

    def fuzzy_score(self, a: str, b: str) -> float:
        """Compute token-sort-ratio fuzzy similarity.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if not a or not b:
            return 0.0
        return fuzz.token_sort_ratio(a, b) / 100.0

    def semantic_score(self, a: str, b: str) -> float:
        """Compute cosine similarity using sentence-BERT embeddings.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Cosine similarity between 0.0 and 1.0.
        """
        if not a or not b:
            return 0.0

        embeddings = self.model.encode([a, b], convert_to_numpy=True)
        vec_a = embeddings[0]
        vec_b = embeddings[1]

        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        cosine_sim = float(dot_product / (norm_a * norm_b))
        return max(0.0, min(1.0, cosine_sim))

    def combined_score(
        self,
        a: str,
        b: str,
        fuzzy_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
    ) -> float:
        """Compute weighted combination of fuzzy and semantic scores.

        Args:
            a: First string.
            b: Second string.
            fuzzy_weight: Override default fuzzy weight.
            semantic_weight: Override default semantic weight.

        Returns:
            Weighted combined score between 0.0 and 1.0.
        """
        fw = fuzzy_weight if fuzzy_weight is not None else self.fuzzy_weight
        sw = semantic_weight if semantic_weight is not None else self.semantic_weight

        f_score = self.fuzzy_score(a, b)
        s_score = self.semantic_score(a, b)

        return fw * f_score + sw * s_score

    def batch_fuzzy_scores(self, query: str, candidates: list[str]) -> list[float]:
        """Compute fuzzy scores for a query against multiple candidates.

        Args:
            query: Query string.
            candidates: List of candidate strings.

        Returns:
            List of fuzzy scores.
        """
        return [self.fuzzy_score(query, c) for c in candidates]

    def batch_semantic_scores(self, query: str, candidates: list[str]) -> list[float]:
        """Compute semantic scores for a query against multiple candidates.

        Args:
            query: Query string.
            candidates: List of candidate strings.

        Returns:
            List of semantic scores.
        """
        if not query or not candidates:
            return [0.0] * len(candidates)

        all_texts = [query] + candidates
        embeddings = self.model.encode(all_texts, convert_to_numpy=True)

        query_vec = embeddings[0]
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return [0.0] * len(candidates)

        scores = []
        for i in range(1, len(embeddings)):
            cand_vec = embeddings[i]
            cand_norm = np.linalg.norm(cand_vec)
            if cand_norm == 0:
                scores.append(0.0)
            else:
                sim = float(np.dot(query_vec, cand_vec) / (query_norm * cand_norm))
                scores.append(max(0.0, min(1.0, sim)))

        return scores
