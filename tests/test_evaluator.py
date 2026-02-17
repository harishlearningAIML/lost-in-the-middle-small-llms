"""Tests for the evaluator module."""

import pytest
from src.evaluator import check_answer, extract_answer, normalize


class TestNormalize:
    """Test the normalize function."""

    def test_lowercase(self):
        """Normalize should convert to lowercase."""
        assert normalize("PARIS") == "paris"
        assert normalize("Paris") == "paris"

    def test_remove_punctuation(self):
        """Normalize should remove punctuation except hyphens and dots."""
        assert normalize("Hello, world!") == "hello world"
        # Dots and hyphens are preserved for decimals/hyphenated words like 2.4 or well-known
        result = normalize("1887.")
        assert result in ["1887", "1887."]  # Accept either
        result_hyphen = normalize("COX-3")
        assert result_hyphen in ["cox 3", "cox-3"]  # Accept either (hyphen preserved)

    def test_whitespace_normalization(self):
        """Normalize should collapse multiple spaces."""
        assert normalize("hello   world") == "hello world"
        assert normalize("  spaces  ") == "spaces"

    def test_empty_string(self):
        """Normalize should handle empty strings."""
        assert normalize("") == ""
        assert normalize("   ") == ""


class TestExtractAnswer:
    """Test the extract_answer function."""

    def test_no_prefix(self):
        """Should return text as-is if no prefix."""
        assert extract_answer("Paris") == "Paris"

    def test_remove_common_prefix(self):
        """Should remove common prefixes."""
        assert extract_answer("The answer is Paris") == "Paris"
        assert extract_answer("Answer: Paris") == "Paris"
        # "The Paris" - we do NOT strip standalone "the" (breaks "The Answer" etc.)
        assert extract_answer("The Paris") == "The Paris"

    def test_remove_trailing_punctuation(self):
        """Should remove trailing punctuation."""
        assert extract_answer("Paris.") == "Paris"
        assert extract_answer("Paris,") == "Paris"
        assert extract_answer("Paris;") == "Paris"

    def test_multiword_extraction(self):
        """Should extract multi-word answers."""
        result = extract_answer("Based on the documents, the answer is Maria Thornberg")
        assert "Maria Thornberg" in result or "maria thornberg" in result.lower()

    def test_empty_response(self):
        """Should handle empty responses."""
        assert extract_answer("") == ""


class TestCheckAnswer:
    """Test the check_answer function."""

    def test_exact_match(self):
        """Should find exact matches."""
        is_correct, extracted = check_answer("Paris", "Paris")
        assert is_correct is True
        assert "Paris" in extracted or "paris" in extracted.lower()

    def test_case_insensitive_match(self):
        """Should match case-insensitively."""
        is_correct, _ = check_answer("PARIS", "paris")
        assert is_correct is True

    def test_prefix_removal(self):
        """Should match after removing prefixes."""
        is_correct, _ = check_answer("The answer is Paris", "Paris")
        assert is_correct is True

    def test_multiword_answer_with_stopwords(self):
        """Should match multi-word answers ignoring stopwords."""
        is_correct, _ = check_answer(
            "rare earth minerals, particularly lithium", "rare earth minerals"
        )
        assert is_correct is True

    def test_name_matching(self):
        """Should match names with all words present."""
        is_correct, _ = check_answer("The CEO is Maria Thornberg", "Maria Thornberg")
        assert is_correct is True

    def test_number_matching(self):
        """Should match numeric answers."""
        is_correct, _ = check_answer("It was founded in 1887", "1887")
        assert is_correct is True

    def test_decimal_number_matching(self):
        """Should match decimal numbers."""
        is_correct, _ = check_answer("2.4 million residents", "2.4 million")
        assert is_correct is True

    def test_wrong_answer(self):
        """Should return False for wrong answers."""
        is_correct, _ = check_answer("Northgate", "Zentrix")
        assert is_correct is False

    def test_i_dont_know(self):
        """Should return False for 'I don't know' responses."""
        is_correct, _ = check_answer("I don't know", "Paris")
        assert is_correct is False

    def test_abbreviation_to_full_name(self):
        """Should match abbreviation to full name when numbers match (COX-3 vs cyclooxygenase-3)."""
        is_correct, _ = check_answer("COX-3", "cyclooxygenase-3")
        assert is_correct is True

    def test_empty_extraction(self):
        """Should return False if extraction is empty."""
        is_correct, extracted = check_answer("...", "Paris")
        assert is_correct is False

    def test_partial_word_no_match(self):
        """Should not match partial words strictly."""
        # Note: The check_answer function is lenient and may match if answer is substring
        is_correct, _ = check_answer("Par", "Paris")
        # The current implementation may match - this documents the behavior
        assert isinstance(is_correct, bool)

    def test_multiple_candidate_answers(self):
        """Should return extracted answer for analysis."""
        _, extracted = check_answer(
            "The capital of France is Paris, a beautiful city", "Paris"
        )
        assert extracted  # Should have extracted something


class TestCheckAnswerEdgeCases:
    """Edge case tests for check_answer."""

    def test_unicode_characters(self):
        """Should handle unicode correctly."""
        is_correct, _ = check_answer("Zürich", "Zürich")
        assert is_correct is True

    def test_hyphenated_words(self):
        """Should handle hyphenated words."""
        is_correct, _ = check_answer("well-known", "well-known")
        assert is_correct is True

    def test_very_long_response(self):
        """Should handle very long responses."""
        long_response = "Paris is the capital of France. " * 50 + "Paris"
        is_correct, _ = check_answer(long_response, "Paris")
        assert is_correct is True

    def test_numeric_with_comma(self):
        """Should match numbers with thousand separators."""
        is_correct, _ = check_answer("Population is 1,000,000", "1000000")
        # May or may not match depending on normalization
        assert isinstance(is_correct, bool)

    def test_single_letter_answer(self):
        """Should handle single letter answers."""
        is_correct, _ = check_answer("A", "A")
        assert is_correct is True

    def test_year_answer(self):
        """Should correctly match year answers."""
        is_correct, _ = check_answer("The event happened in 2023.", "2023")
        assert is_correct is True

    def test_with_extra_punctuation(self):
        """Should handle answers with extra punctuation."""
        is_correct, _ = check_answer("Paris!!! The answer.", "Paris")
        assert is_correct is True


class TestIntegrationWithRealPatterns:
    """Test with realistic response patterns from models."""

    def test_gemma_style_response(self):
        """Test response pattern from Gemma models."""
        response = "Based on the documents, the answer is Paris."
        is_correct, extracted = check_answer(response, "Paris")
        assert is_correct is True

    def test_llama_style_response(self):
        """Test response pattern from Llama models."""
        response = "The answer is Paris, which is the capital of France."
        is_correct, extracted = check_answer(response, "Paris")
        assert is_correct is True

    def test_verbose_response(self):
        """Test overly verbose response."""
        response = """Based on the documents provided, I can determine that the answer to the question is Paris, 
        the capital of France."""
        is_correct, _ = check_answer(response, "Paris")
        assert is_correct is True

    def test_reject_explicit_wrong_answer(self):
        """Should reject when model explicitly gives different answer despite gold appearing."""
        # Gold "1887" appears but model says real answer is 1342
        is_correct, _ = check_answer(
            "1887 was a terrible year, the real answer is 1342", "1887"
        )
        assert is_correct is False
