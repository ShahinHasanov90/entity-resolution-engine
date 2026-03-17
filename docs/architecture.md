# Architecture

## System Overview

The Multilingual Entity Resolver uses a two-stage pipeline for company name resolution across Azerbaijani, Russian, and English text.

## Pipeline Stages

```
Input Query
    │
    ▼
┌───────────────────┐
│  Preprocessor     │  Unicode normalization, stopword removal,
│  (normalize)      │  transliteration, case folding
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Cache Lookup     │  TTL-based LRU cache check
│                   │  Hit → return cached result
└───────┬───────────┘
        │ Miss
        ▼
┌───────────────────┐
│  Stage 1: Fuzzy   │  rapidfuzz token_sort_ratio
│  Candidate Select │  Select top-K candidates (default K=10)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Stage 2: Semantic│  sentence-BERT (all-MiniLM-L6-v2)
│  Re-ranking       │  Cosine similarity re-ranking
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Score Combination│  weighted: 0.4×fuzzy + 0.6×semantic
│  & Threshold      │  Filter by threshold (default 0.85)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Cache Store      │  Store result for future queries
└───────┬───────────┘
        │
        ▼
    Ranked Results
```

## Component Responsibilities

| Component       | File              | Purpose                                       |
|-----------------|-------------------|-----------------------------------------------|
| Preprocessor    | `preprocessor.py` | Text normalization, transliteration            |
| SimilarityScorer| `similarity.py`   | Fuzzy + semantic scoring                       |
| TTLCache        | `cache.py`        | Thread-safe caching with TTL                   |
| CompanyResolver | `resolver.py`     | Orchestrates the full resolution pipeline      |
| API             | `api.py`          | FastAPI HTTP interface                         |

## Design Decisions

1. **Two-stage approach**: Fuzzy matching is fast but misses semantic relationships; BERT is accurate but slow. The combination gives both speed and accuracy.

2. **Transliteration before matching**: Converting Cyrillic to Latin allows cross-script matching without needing multilingual embeddings.

3. **LRU + TTL cache**: Customs documents often repeat the same companies. Caching dramatically reduces latency for repeated queries while TTL ensures freshness.

4. **Configurable weights**: Different deployment contexts may need different fuzzy/semantic balances. YAML configuration allows tuning without code changes.
