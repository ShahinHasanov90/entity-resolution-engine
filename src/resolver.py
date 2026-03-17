"""Main CompanyResolver class — two-stage entity resolution pipeline.

Stage 1: rapidfuzz token_sort_ratio selects top-K candidates.
Stage 2: sentence-BERT re-ranks candidates by semantic similarity.
Combined score is used for final ranking and threshold filtering.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .cache import TTLCache
from .preprocessor import normalize_company_name
from .similarity import SimilarityScorer

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """A single match result from the resolver.

    Attributes:
        name: Canonical company name.
        matched_variant: The specific variant that matched.
        score: Combined similarity score (0-1).
        fuzzy_score: Token-based fuzzy score component.
        semantic_score: Embedding-based semantic score component.
        method: Description of the matching method used.
        company_id: Optional identifier from the company registry.
    """
    name: str
    matched_variant: str
    score: float
    fuzzy_score: float
    semantic_score: float
    method: str
    company_id: Optional[int] = None


@dataclass
class CompanyEntry:
    """A registered company with its known name variants.

    Attributes:
        id: Unique company identifier.
        canonical: Canonical (official) company name.
        variants: List of known name variants across languages.
    """
    id: int
    canonical: str
    variants: list[str] = field(default_factory=list)

    @property
    def all_names(self) -> list[str]:
        """All searchable names including canonical."""
        return [self.canonical] + self.variants


class CompanyResolver:
    """Two-stage company name resolver using fuzzy matching + semantic similarity.

    The resolution pipeline:
        1. Normalize the input query.
        2. Check the cache for a previous result.
        3. Compute fuzzy scores against all known variants.
        4. Select top-K candidates by fuzzy score.
        5. Re-rank candidates using sentence-BERT semantic similarity.
        6. Return results above the combined score threshold.

    Args:
        scorer: SimilarityScorer instance.
        threshold: Minimum combined score to accept a match.
        fuzzy_top_k: Number of candidates to pass from fuzzy to semantic stage.
        cache_max_size: Maximum cache entries.
        cache_ttl: Cache TTL in seconds.
    """

    def __init__(
        self,
        scorer: Optional[SimilarityScorer] = None,
        threshold: float = 0.85,
        fuzzy_top_k: int = 10,
        cache_max_size: int = 10000,
        cache_ttl: float = 3600.0,
    ) -> None:
        self.scorer = scorer or SimilarityScorer()
        self.threshold = threshold
        self.fuzzy_top_k = fuzzy_top_k
        self.cache = TTLCache(max_size=cache_max_size, ttl_seconds=cache_ttl)
        self._companies: list[CompanyEntry] = []
        self._variant_index: dict[str, tuple[int, str]] = {}

    def load_companies(self, data_path: str | Path) -> int:
        """Load company entries from a JSON file.

        Args:
            data_path: Path to JSON file with company entries.

        Returns:
            Number of companies loaded.

        Raises:
            FileNotFoundError: If data file does not exist.
            json.JSONDecodeError: If data file is invalid JSON.
        """
        data_path = Path(data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self._companies.clear()
        self._variant_index.clear()

        for entry in raw_data:
            company = CompanyEntry(
                id=entry["id"],
                canonical=entry["canonical"],
                variants=entry.get("variants", []),
            )
            self._companies.append(company)

            for name in company.all_names:
                normalized = normalize_company_name(name)
                self._variant_index[normalized] = (company.id, name)

        logger.info(
            "Loaded %d companies with %d total variants",
            len(self._companies),
            len(self._variant_index),
        )
        return len(self._companies)

    def add_company(self, company: CompanyEntry) -> None:
        """Register a single company entry.

        Args:
            company: CompanyEntry to add.
        """
        self._companies.append(company)
        for name in company.all_names:
            normalized = normalize_company_name(name)
            self._variant_index[normalized] = (company.id, name)

    def resolve(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> list[MatchResult]:
        """Resolve a company name to known entities.

        Args:
            query: Input company name to resolve.
            top_k: Maximum number of results to return.
            threshold: Override default threshold for this query.

        Returns:
            List of MatchResult sorted by descending score.
        """
        if not query or not query.strip():
            return []

        effective_threshold = threshold if threshold is not None else self.threshold
        normalized_query = normalize_company_name(query)

        if not normalized_query:
            return []

        # Check cache
        cache_key = f"{normalized_query}:{top_k}:{effective_threshold}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Stage 1: Fuzzy matching to select candidates
        all_variants = list(self._variant_index.keys())
        if not all_variants:
            return []

        fuzzy_scores = self.scorer.batch_fuzzy_scores(normalized_query, all_variants)

        # Select top-K by fuzzy score
        scored_variants = list(zip(all_variants, fuzzy_scores))
        scored_variants.sort(key=lambda x: x[1], reverse=True)
        candidates = scored_variants[: self.fuzzy_top_k]

        # Stage 2: Semantic re-ranking
        candidate_names = [c[0] for c in candidates]
        candidate_fuzzy = {c[0]: c[1] for c in candidates}
        semantic_scores = self.scorer.batch_semantic_scores(
            normalized_query, candidate_names
        )

        # Combine scores and build results
        results: list[MatchResult] = []
        for variant_norm, sem_score in zip(candidate_names, semantic_scores):
            f_score = candidate_fuzzy[variant_norm]
            combined = (
                self.scorer.fuzzy_weight * f_score
                + self.scorer.semantic_weight * sem_score
            )

            if combined >= effective_threshold:
                company_id, original_name = self._variant_index[variant_norm]
                company = next(
                    (c for c in self._companies if c.id == company_id), None
                )
                canonical = company.canonical if company else original_name

                results.append(
                    MatchResult(
                        name=canonical,
                        matched_variant=original_name,
                        score=round(combined, 4),
                        fuzzy_score=round(f_score, 4),
                        semantic_score=round(sem_score, 4),
                        method="fuzzy+semantic",
                        company_id=company_id,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:top_k]

        # Cache results
        self.cache.put(cache_key, results)

        return results

    @property
    def company_count(self) -> int:
        """Number of registered companies."""
        return len(self._companies)

    @property
    def variant_count(self) -> int:
        """Number of indexed name variants."""
        return len(self._variant_index)
