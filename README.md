# Multilingual Entity Resolver

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![NLP](https://img.shields.io/badge/NLP-sentence--BERT-orange)

Production-ready NLP pipeline for resolving company names across **Azerbaijani**, **Russian**, and **English** in customs trade documents using fuzzy matching and semantic similarity.

## Problem Statement

International trade documents — customs declarations, bills of lading, transit permits — frequently reference the same company under different names, scripts, and transliterations. A single entity like "Qafqaz Dəmir Yolları" might appear as "Caucasus Railways", "КДЖ", or "Кавказские Железные Дороги" across different records. Manual reconciliation is slow, error-prone, and doesn't scale. This pipeline automates multilingual entity resolution with sub-second latency.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Input Query │────▶│ Preprocessor │────▶│  Cache Lookup    │
│  (AZ/RU/EN)  │     │  • Unicode   │     │  • LRU + TTL     │
│              │     │  • Stopwords │     │  • Thread-safe   │
└──────────────┘     │  • Translit  │     └────────┬─────────┘
                     └──────────────┘              │
                                          ┌────────▼─────────┐
                                          │  Stage 1: Fuzzy  │
                                          │  rapidfuzz top-K │
                                          │  (token_sort)    │
                                          └────────┬─────────┘
                                          ┌────────▼─────────┐
                                          │  Stage 2: BERT   │
                                          │  Semantic re-rank│
                                          │  (all-MiniLM-L6) │
                                          └────────┬─────────┘
                                          ┌────────▼─────────┐
                                          │  Combined Score  │
                                          │  0.4×fuzzy +     │
                                          │  0.6×semantic    │
                                          └────────┬─────────┘
                                                   ▼
                                          Ranked Results (JSON)
```

## Features

- **Two-stage resolution**: Fast fuzzy candidate selection → accurate semantic re-ranking
- **Trilingual support**: Azerbaijani, Russian, and English with cross-script matching
- **Transliteration engine**: Cyrillic ↔ Latin automatic conversion (AZ/RU mappings)
- **Company-type stopword removal**: Handles LLC, MMC, ООО, ASC, QSC, ŞTH, and more
- **Thread-safe LRU cache**: Configurable TTL and max size for high-throughput scenarios
- **REST API**: FastAPI-based service with OpenAPI documentation
- **Configurable thresholds**: YAML-driven weights, thresholds, and model selection

## Quick Start

### Installation

```bash
git clone https://github.com/ShahinHasanov90/multilingual-entity-resolver.git
cd multilingual-entity-resolver
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The sentence-BERT model downloads automatically on first startup (~90 MB).

### Example Request

```bash
curl -X POST http://localhost:8000/resolve \
  -H "Content-Type: application/json" \
  -d '{"company_name": "Кавказские Железные Дороги", "top_k": 3}'
```

### Example Response

```json
{
  "query": "Кавказские Железные Дороги",
  "results": [
    {
      "name": "Qafqaz Dəmir Yolları",
      "matched_variant": "Кавказские Железные Дороги",
      "score": 0.9712,
      "fuzzy_score": 0.9500,
      "semantic_score": 0.9854,
      "method": "fuzzy+semantic",
      "company_id": 1
    }
  ],
  "count": 1,
  "elapsed_ms": 12.34
}
```

## Configuration

Edit `config/settings.yaml`:

```yaml
resolver:
  match_threshold: 0.85    # Minimum combined score to accept
  fuzzy_top_k: 10          # Candidates from fuzzy stage

similarity:
  fuzzy_weight: 0.4        # Token-based matching weight
  semantic_weight: 0.6     # Embedding similarity weight
  model_name: "all-MiniLM-L6-v2"

cache:
  max_size: 10000
  ttl_seconds: 3600
```

## Performance

| Metric | Value |
|--------|-------|
| Single lookup latency | ~12ms (cached), ~45ms (uncached) |
| Throughput | ~10K lookups/sec on single core (cached) |
| Model load time | ~2s (first request) |
| Memory footprint | ~250MB (model + cache) |

*Benchmarks on synthetic data, single-core, Apple M1.*

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/resolve` | POST | Resolve a company name. Body: `{"company_name": "...", "top_k": 5}` |
| `/health` | GET | Health check with resolver status and cache stats |

Full OpenAPI docs available at `http://localhost:8000/docs` after starting the server.

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Project Structure

```
multilingual-entity-resolver/
├── config/settings.yaml         # All configuration
├── src/
│   ├── resolver.py              # Main CompanyResolver class
│   ├── preprocessor.py          # Text normalization (AZ/RU/EN)
│   ├── similarity.py            # rapidfuzz + sentence-BERT scoring
│   ├── cache.py                 # Thread-safe LRU cache with TTL
│   └── api.py                   # FastAPI endpoints
├── tests/                       # pytest test suite
├── data/sample_companies.json   # 50 synthetic company entries
└── docs/architecture.md         # Detailed architecture docs
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a pull request

## License

MIT License — see [LICENSE](LICENSE) for details.
