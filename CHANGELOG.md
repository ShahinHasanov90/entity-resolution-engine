# Changelog

## [1.1.0] - 2026-01-05
### Added
- Thread-safe LRU cache with TTL for high-throughput scenarios
- Company-type stopword removal (LLC, MMC, OOO, ASC, QSC)

### Changed
- Tuned similarity weights: 0.4 fuzzy + 0.6 semantic

## [1.0.0] - 2025-07-01
### Added
- Initial release: Two-stage resolution (fuzzy + semantic)
- Trilingual support (AZ/RU/EN) with cross-script matching
- FastAPI REST service
- Cyrillic-Latin transliteration engine
