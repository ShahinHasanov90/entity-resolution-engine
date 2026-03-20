"""Tests for the main CompanyResolver class."""

import json
import tempfile
from pathlib import Path

import pytest

from src.resolver import CompanyEntry, CompanyResolver, MatchResult
from src.similarity import SimilarityScorer


@pytest.fixture(scope="module")
def sample_data_path() -> Path:
    """Create a temporary company data file for testing."""
    data = [
        {
            "id": 1,
            "canonical": "Caucasus Railways",
            "variants": ["Qafqaz Dəmir Yolları", "Кавказские Железные Дороги", "QDY"],
        },
        {
            "id": 2,
            "canonical": "Atlas Logistics",
            "variants": ["Atlas Logistika MMC", "Атлас Логистика ООО"],
        },
        {
            "id": 3,
            "canonical": "Silk Road Cargo",
            "variants": ["İpək Yolu Daşımaçılıq", "Шёлковый Путь Грузоперевозки"],
        },
        {
            "id": 4,
            "canonical": "Caspian Maritime Company",
            "variants": ["Xəzər Dəniz Şirkəti", "Каспийская Морская Компания", "CMC"],
        },
        {
            "id": 5,
            "canonical": "Capital Transport Agency",
            "variants": ["Paytaxt Nəqliyyat Agentliyi", "Столичное Транспортное Агентство", "CTA"],
        },
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(data, f, ensure_ascii=False)
        return Path(f.name)


@pytest.fixture(scope="module")
def resolver(sample_data_path: Path) -> CompanyResolver:
    """Resolver instance with loaded test data."""
    scorer = SimilarityScorer(
        model_name="all-MiniLM-L6-v2",
        fuzzy_weight=0.4,
        semantic_weight=0.6,
    )
    scorer.load_model()

    r = CompanyResolver(
        scorer=scorer,
        threshold=0.5,  # Lower threshold for testing
        fuzzy_top_k=10,
    )
    r.load_companies(sample_data_path)
    return r


class TestCompanyResolverLoad:
    def test_load_count(self, resolver: CompanyResolver) -> None:
        assert resolver.company_count == 5

    def test_variant_count(self, resolver: CompanyResolver) -> None:
        # 5 companies * (1 canonical + variants each)
        assert resolver.variant_count > 5


class TestResolveExactMatch:
    def test_exact_canonical(self, resolver: CompanyResolver) -> None:
        results = resolver.resolve("Caucasus Railways")
        assert len(results) > 0
        assert results[0].name == "Caucasus Railways"
        assert results[0].score > 0.9

    def test_exact_variant(self, resolver: CompanyResolver) -> None:
        results = resolver.resolve("Atlas Logistika MMC")
        assert len(results) > 0
        assert results[0].name == "Atlas Logistics"


class TestResolveFuzzyMatch:
    def test_typo_tolerance(self, resolver: CompanyResolver) -> None:
        results = resolver.resolve("Caucasus Railwayss")
        assert len(results) > 0
        assert results[0].name == "Caucasus Railways"

    def test_partial_name(self, resolver: CompanyResolver) -> None:
        results = resolver.resolve("Silk Road")
        assert len(results) > 0
        assert "Silk Road" in results[0].name


class TestResolveCrossLanguage:
    def test_russian_to_canonical(self, resolver: CompanyResolver) -> None:
        results = resolver.resolve("Атлас Логистика")
        assert len(results) > 0
        assert results[0].name == "Atlas Logistics"

    def test_azerbaijani_to_canonical(self, resolver: CompanyResolver) -> None:
        results = resolver.resolve("Qafqaz Dəmir Yolları")
        assert len(results) > 0
        assert results[0].name == "Caucasus Railways"


class TestResolveNoMatch:
    def test_completely_unrelated(self, resolver: CompanyResolver) -> None:
        r = CompanyResolver(
            scorer=resolver.scorer,
            threshold=0.99,  # Very high threshold
        )
        r.load_companies(Path(tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ).name).parent / "nonexistent.json") if False else None
        # Test with high threshold on main resolver
        results = resolver.resolve("XYZQWERTY Nonsense Corp", threshold=0.99)
        assert len(results) == 0

    def test_empty_query(self, resolver: CompanyResolver) -> None:
        assert resolver.resolve("") == []

    def test_whitespace_query(self, resolver: CompanyResolver) -> None:
        assert resolver.resolve("   ") == []


class TestResolveTopK:
    def test_top_k_limit(self, resolver: CompanyResolver) -> None:
        results = resolver.resolve("Azerbaijan", top_k=2)
        assert len(results) <= 2

    def test_top_k_one(self, resolver: CompanyResolver) -> None:
        results = resolver.resolve("Azerbaijan Railways", top_k=1)
        assert len(results) <= 1


class TestResolveCache:
    def test_cache_hit(self, resolver: CompanyResolver) -> None:
        # First call populates cache
        results1 = resolver.resolve("Silk Way Airlines")
        # Second call should hit cache
        results2 = resolver.resolve("Silk Way Airlines")
        assert len(results1) == len(results2)
        assert results1[0].score == results2[0].score


class TestMatchResult:
    def test_result_fields(self, resolver: CompanyResolver) -> None:
        results = resolver.resolve("Caucasus Railways")
        if results:
            r = results[0]
            assert isinstance(r.name, str)
            assert isinstance(r.score, float)
            assert isinstance(r.fuzzy_score, float)
            assert isinstance(r.semantic_score, float)
            assert r.method == "fuzzy+semantic"
            assert r.company_id is not None
