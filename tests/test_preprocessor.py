"""Tests for the text preprocessing / normalization module."""

import pytest

from src.preprocessor import (
    detect_script,
    fold_case,
    normalize_company_name,
    normalize_punctuation,
    normalize_unicode,
    remove_stopwords,
    transliterate_ru_to_latin,
)


class TestNormalizeUnicode:
    def test_nfkd_decomposition(self) -> None:
        result = normalize_unicode("café")
        assert "e" in result  # decomposed form

    def test_passthrough_ascii(self) -> None:
        assert normalize_unicode("hello") == "hello"


class TestFoldCase:
    def test_lowercase(self) -> None:
        assert fold_case("HELLO") == "hello"

    def test_azerbaijani_case(self) -> None:
        result = fold_case("İSTANBUL")
        assert result == result.casefold()


class TestRemoveStopwords:
    def test_remove_llc(self) -> None:
        result = remove_stopwords("Acme LLC")
        assert "LLC" not in result
        assert "Acme" in result

    def test_remove_mmc(self) -> None:
        result = remove_stopwords("Atlas MMC")
        assert "MMC" not in result

    def test_remove_russian_ooo(self) -> None:
        result = remove_stopwords("Атлас ООО")
        assert "ООО" not in result

    def test_preserve_normal_words(self) -> None:
        result = remove_stopwords("Silk Road Cargo")
        assert result == "Silk Road Cargo"


class TestTransliterateRuToLatin:
    def test_basic_cyrillic(self) -> None:
        assert transliterate_ru_to_latin("Москва") == "Moskva"

    def test_complex_name(self) -> None:
        result = transliterate_ru_to_latin("Азербайджан")
        assert result == "Azerbaydzhan"


class TestDetectScript:
    def test_latin(self) -> None:
        assert detect_script("Hello World") == "latin"

    def test_cyrillic(self) -> None:
        assert detect_script("Москва") == "cyrillic"

    def test_mixed(self) -> None:
        assert detect_script("Atlas Трейдинг") == "mixed"


class TestNormalizePunctuation:
    def test_remove_commas(self) -> None:
        assert normalize_punctuation("Acme, Inc.") == "Acme Inc"

    def test_collapse_whitespace(self) -> None:
        assert normalize_punctuation("Acme   Corp") == "Acme Corp"


class TestNormalizeCompanyName:
    def test_full_pipeline_english(self) -> None:
        result = normalize_company_name("Acme Corporation LLC")
        assert "llc" not in result
        assert "acme" in result

    def test_full_pipeline_russian(self) -> None:
        result = normalize_company_name("Атлас Логистика ООО")
        assert "ооо" not in result.lower()
        # Should be transliterated to Latin
        assert "atlas" in result.lower()

    def test_empty_string(self) -> None:
        assert normalize_company_name("") == ""

    def test_whitespace_only(self) -> None:
        assert normalize_company_name("   ") == ""

    def test_azerbaijani_company(self) -> None:
        result = normalize_company_name("Günəş Ticarət MMC")
        assert "mmc" not in result
        assert len(result) > 0

    def test_no_transliteration(self) -> None:
        result = normalize_company_name(
            "Атлас ООО", transliterate_to_latin=False
        )
        # Should stay in Cyrillic
        assert any("\u0400" <= c <= "\u04ff" for c in result)
