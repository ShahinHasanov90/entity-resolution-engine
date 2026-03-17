"""Text normalization for multilingual company names (AZ/RU/EN).

Handles Unicode normalization, case folding, stopword removal,
transliteration mappings, and punctuation cleanup.
"""

import re
import unicodedata
from typing import Optional

# AZ ↔ RU common transliteration mappings
AZ_TO_RU_MAP: dict[str, str] = {
    "Ş": "Ш", "ş": "ш",
    "Ç": "Ч", "ç": "ч",
    "Ğ": "Г", "ğ": "г",
    "I": "И", "ı": "и",
    "İ": "И", "i": "и",
    "Ö": "О", "ö": "о",
    "Ü": "У", "ü": "у",
    "Ə": "Э", "ə": "э",
    "J": "Дж", "j": "дж",
}

RU_TO_LATIN_MAP: dict[str, str] = {
    "А": "A", "а": "a",
    "Б": "B", "б": "b",
    "В": "V", "в": "v",
    "Г": "G", "г": "g",
    "Д": "D", "д": "d",
    "Е": "E", "е": "e",
    "Ё": "E", "ё": "e",
    "Ж": "Zh", "ж": "zh",
    "З": "Z", "з": "z",
    "И": "I", "и": "i",
    "Й": "Y", "й": "y",
    "К": "K", "к": "k",
    "Л": "L", "л": "l",
    "М": "M", "м": "m",
    "Н": "N", "н": "n",
    "О": "O", "о": "o",
    "П": "P", "п": "p",
    "Р": "R", "р": "r",
    "С": "S", "с": "s",
    "Т": "T", "т": "t",
    "У": "U", "у": "u",
    "Ф": "F", "ф": "f",
    "Х": "Kh", "х": "kh",
    "Ц": "Ts", "ц": "ts",
    "Ч": "Ch", "ч": "ch",
    "Ш": "Sh", "ш": "sh",
    "Щ": "Shch", "щ": "shch",
    "Ъ": "", "ъ": "",
    "Ы": "Y", "ы": "y",
    "Ь": "", "ь": "",
    "Э": "E", "э": "e",
    "Ю": "Yu", "ю": "yu",
    "Я": "Ya", "я": "ya",
}

# Default company-type stopwords across AZ/RU/EN
DEFAULT_STOPWORDS: set[str] = {
    # English
    "LLC", "LTD", "INC", "CORP", "CO", "COMPANY", "LIMITED", "CORPORATION",
    "INCORPORATED", "PLC", "PTY", "GMBH", "AG", "SA", "BV", "NV",
    # Azerbaijani
    "MMC", "ASC", "QSC", "ŞTH", "CƏMIYYƏTI", "MƏHDUD", "MƏSULIYYƏTLI",
    "AÇIQ", "SƏHMDAR", "QAPALI",
    # Russian
    "ООО", "ОАО", "ЗАО", "АО", "ИП", "ОБЩЕСТВО", "ОГРАНИЧЕННОЙ",
    "ОТВЕТСТВЕННОСТЬЮ", "АКЦИОНЕРНОЕ", "ЗАКРЫТОЕ", "ОТКРЫТОЕ",
}


def normalize_unicode(text: str) -> str:
    """Apply NFKD Unicode normalization."""
    return unicodedata.normalize("NFKD", text)


def fold_case(text: str) -> str:
    """Apply Unicode case folding (locale-independent lowercase)."""
    return text.casefold()


def remove_stopwords(text: str, extra_stopwords: Optional[set[str]] = None) -> str:
    """Remove company-type stopwords from text.

    Args:
        text: Input text (should be uppercase or original case for matching).
        extra_stopwords: Additional stopwords to remove beyond defaults.

    Returns:
        Text with stopwords removed.
    """
    stopwords = DEFAULT_STOPWORDS.copy()
    if extra_stopwords:
        stopwords.update(extra_stopwords)

    tokens = text.split()
    filtered = [t for t in tokens if t.upper().strip(".,()-\"'") not in stopwords]
    return " ".join(filtered)


def transliterate_az_to_ru(text: str) -> str:
    """Transliterate Azerbaijani characters to Russian equivalents."""
    for az_char, ru_char in AZ_TO_RU_MAP.items():
        text = text.replace(az_char, ru_char)
    return text


def transliterate_ru_to_latin(text: str) -> str:
    """Transliterate Russian Cyrillic characters to Latin equivalents."""
    result = []
    for char in text:
        result.append(RU_TO_LATIN_MAP.get(char, char))
    return "".join(result)


def normalize_punctuation(text: str) -> str:
    """Remove punctuation and normalize whitespace."""
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_script(text: str) -> str:
    """Detect the dominant script in text.

    Returns:
        One of 'cyrillic', 'latin', or 'mixed'.
    """
    cyrillic_count = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    latin_count = sum(1 for c in text if ("A" <= c <= "Z") or ("a" <= c <= "z"))

    if cyrillic_count > latin_count:
        return "cyrillic"
    elif latin_count > cyrillic_count:
        return "latin"
    return "mixed"


def normalize_company_name(
    name: str,
    remove_company_types: bool = True,
    transliterate_to_latin: bool = True,
    extra_stopwords: Optional[set[str]] = None,
) -> str:
    """Full normalization pipeline for a company name.

    Steps:
        1. Unicode normalization (NFKD)
        2. Stopword removal (company types in AZ/RU/EN)
        3. Optional transliteration to Latin script
        4. Case folding
        5. Punctuation and whitespace normalization

    Args:
        name: Raw company name in any supported language.
        remove_company_types: Whether to strip legal entity suffixes.
        transliterate_to_latin: Whether to convert Cyrillic to Latin.
        extra_stopwords: Additional stopwords beyond the defaults.

    Returns:
        Normalized company name string.
    """
    if not name or not name.strip():
        return ""

    text = normalize_unicode(name)

    if remove_company_types:
        text = remove_stopwords(text, extra_stopwords)

    if transliterate_to_latin:
        script = detect_script(text)
        if script == "cyrillic":
            text = transliterate_ru_to_latin(text)
        elif script == "mixed":
            text = transliterate_ru_to_latin(text)

    text = fold_case(text)
    text = normalize_punctuation(text)

    return text
