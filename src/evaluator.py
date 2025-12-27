"""Evaluate model responses against gold answers."""

import re
from typing import Tuple


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase
    text = text.lower()
    # Remove punctuation except hyphens in numbers
    text = re.sub(r'[^\w\s\-\.]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()


def extract_answer(response: str) -> str:
    """
    Extract the core answer from model response.
    Handles common response patterns.
    """
    # Remove common prefixes
    prefixes = [
        "the answer is",
        "answer:",
        "the",
        "based on the documents,",
        "according to the documents,",
    ]
    
    text = response.strip()
    text_lower = text.lower()
    
    for prefix in prefixes:
        if text_lower.startswith(prefix):
            text = text[len(prefix):].strip()
            text_lower = text.lower()
    
    # Remove trailing punctuation
    text = text.rstrip('.,;:')
    
    return text


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
    
    # Normalize both for comparison
    norm_extracted = normalize(extracted)
    norm_gold = normalize(gold_answer)
    
    # Exact match (but not if extracted is empty)
    if norm_extracted and (norm_gold in norm_extracted or norm_extracted in norm_gold):
        return True, extracted
    
    # Check if gold answer words are present
    gold_words = set(norm_gold.split())
    extracted_words = set(norm_extracted.split())
    
    # If all significant gold words are in extracted (for multi-word answers)
    if len(gold_words) > 1:
        # Remove common words
        stopwords = {'the', 'a', 'an', 'of', 'and', 'in', 'is', 'was', 'are', 'were'}
        gold_significant = gold_words - stopwords
        if gold_significant and gold_significant.issubset(extracted_words):
            return True, extracted
    
    # Check for key numeric/name matches
    # Extract numbers from both
    gold_numbers = set(re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', norm_gold))
    extracted_numbers = set(re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', norm_extracted))
    
    if gold_numbers and gold_numbers.issubset(extracted_numbers):
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
        print(f"{status} '{response}' vs '{gold}' -> {is_correct} (extracted: '{extracted}')")
