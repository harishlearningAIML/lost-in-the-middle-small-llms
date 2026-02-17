"""Evaluate model responses against gold answers."""

import re
from typing import Optional, Tuple


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase
    text = text.lower()
    # Remove punctuation except hyphens in numbers
    text = re.sub(r"[^\w\s\-\.]", "", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()


def extract_answer(response: str) -> str:
    """
    Extract the core answer from model response.
    Handles common response patterns.
    """
    # Remove common prefixes (do NOT use standalone "the" - breaks "The Answer" etc.)
    prefixes = [
        "the answer is",
        "answer:",
        "based on the documents,",
        "according to the documents,",
    ]

    text = response.strip()
    text_lower = text.lower()

    for prefix in prefixes:
        if text_lower.startswith(prefix):
            text = text[len(prefix) :].strip()
            text_lower = text.lower()

    # Remove trailing punctuation
    text = text.rstrip(".,;:")

    return text


def _extract_explicit_answer(text: str) -> Optional[str]:
    """
    If the model explicitly states an answer (e.g. 'the real answer is X'),
    extract it. Used to reject false positives when gold appears but wrong answer given.
    """
    text_lower = text.lower()
    patterns = [
        r"(?:the\s+)?real\s+answer\s+is\s+([^.,]+)",
        r"(?:the\s+)?(?:correct\s+)?answer\s+is\s+([^.,]+)",
        r"answer\s*:\s*([^.,]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text_lower, re.IGNORECASE)
        if m:
            return normalize(m.group(1))
    return None


def check_answer(response: str, gold_answer: str) -> Tuple[bool, str]:
    """
    Check if the response contains the gold answer.

    Args:
        response: Model's response
        gold_answer: Expected answer

    Returns:
        Tuple of (is_correct, extracted_answer)
    """
    extracted = extract_answer(response)

    # Reject if model explicitly gives a different answer
    explicit = _extract_explicit_answer(response)
    if explicit:
        norm_gold = normalize(gold_answer)
        if explicit != norm_gold and norm_gold not in explicit and explicit not in norm_gold:
            return False, extracted

    # Normalize both for comparison
    norm_extracted = normalize(extracted)
    norm_gold = normalize(gold_answer)

    # 1. Strict exact match after normalization
    if norm_extracted == norm_gold:
        return True, extracted

    # 2. Check if all significant gold words are present in extracted (for multi-word answers)
    gold_words = set(norm_gold.split())
    extracted_words = set(norm_extracted.split())

    if len(gold_words) > 1:
        # Remove common words (stopwords logic as before)
        stopwords = {"the", "a", "an", "of", "and", "in", "is", "was", "are", "were"}
        gold_significant = gold_words - stopwords
        if gold_significant and gold_significant.issubset(extracted_words):
            return True, extracted

    # 3. Check for numeric matches (stricter than before)
    gold_numbers = set(re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", norm_gold))
    extracted_numbers = set(re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", norm_extracted))

    if gold_numbers:
        # Require an exact set match of numbers
        if gold_numbers == extracted_numbers:
            return True, extracted

    return False, extracted


if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("Zentrix", "Zentrix", True),
        ("The capital is Zentrix.", "Zentrix", True),
        ("Maria Thornberg", "Maria Thornberg", True),
        ("The CEO is Maria Thornberg since 2023.", "Maria Thornberg", True),
        ("1887", "1887", True),
        ("It was founded in 1887.", "1887", True),
        ("rare earth minerals, particularly lithium", "rare earth minerals", True),
        ("2.4 million residents", "2.4 million", True),
        ("cyclooxygenase-3 (COX-3)", "cyclooxygenase-3", True),
        ("COX-3", "cyclooxygenase-3", True),  # Abbreviation match
        ("Northgate", "Zentrix", False),  # Wrong answer
        ("I don't know", "Zentrix", False),
    ]

    print("Testing evaluator:")
    for response, gold, expected in test_cases:
        is_correct, extracted = check_answer(response, gold)
        status = "✓" if is_correct == expected else "✗"
        print(
            f"{status} '{response}' vs '{gold}' -> {is_correct} (extracted: '{extracted}')"
        )
