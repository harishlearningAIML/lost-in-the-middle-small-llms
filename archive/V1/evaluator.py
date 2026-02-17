"""
Evaluator - Check if model output contains correct answer
"""

import re
from typing import List, Tuple


def normalize(text: str) -> str:
    """Normalize text for comparison"""
    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_answer(model_output: str) -> str:
    """
    Extract the answer from model output.
    Models might include extra text, so we try to get just the answer.
    """
    # Clean up the output
    answer = model_output.strip()
    
    # Remove common prefixes
    prefixes = [
        "the answer is",
        "answer:",
        "the capital is",
        "it is",
        "it's",
    ]
    
    lower = answer.lower()
    for prefix in prefixes:
        if lower.startswith(prefix):
            answer = answer[len(prefix):].strip()
            break
    
    # Take first line/sentence if multiple
    answer = answer.split('\n')[0]
    answer = answer.split('.')[0]
    
    return answer.strip()


def check_answer(
    model_output: str,
    gold_answer: str,
    answer_variants: List[str] = None
) -> Tuple[bool, str]:
    """
    Check if model output contains the correct answer.
    
    Args:
        model_output: Raw model output
        gold_answer: The correct answer
        answer_variants: Alternative acceptable answers
    
    Returns:
        (is_correct, extracted_answer)
    """
    extracted = extract_answer(model_output)
    normalized_output = normalize(model_output)
    normalized_extracted = normalize(extracted)
    
    # Check against gold answer
    normalized_gold = normalize(gold_answer)
    
    # Build list of acceptable answers
    acceptable = [normalized_gold]
    if answer_variants:
        acceptable.extend([normalize(v) for v in answer_variants])
    
    # Check if any acceptable answer is in the output
    for acceptable_answer in acceptable:
        # Exact match on extracted
        if normalized_extracted == acceptable_answer:
            return True, extracted
        
        # Contained in extracted (for partial matches)
        if acceptable_answer in normalized_extracted:
            return True, extracted
        
        # Contained in full output
        if acceptable_answer in normalized_output:
            return True, extracted
    
    return False, extracted


# Test
if __name__ == "__main__":
    test_cases = [
        # (model_output, gold_answer, variants, expected)
        ("Zentrix", "Zentrix", ["zentrix"], True),
        ("The capital is Zentrix.", "Zentrix", None, True),
        ("I believe the answer is Zentrix, a historic city.", "Zentrix", None, True),
        ("Hillford", "Zentrix", None, False),
        ("Maria Thornberg", "Maria Thornberg", ["Thornberg"], True),
        ("The CEO is Thornberg.", "Maria Thornberg", ["Thornberg"], True),
        ("2.4 million residents", "2.4 million", ["2,400,000"], True),
        ("1887", "1887", None, True),
    ]
    
    print("Testing evaluator...")
    print("-" * 60)
    
    for output, gold, variants, expected in test_cases:
        is_correct, extracted = check_answer(output, gold, variants)
        status = "✓" if is_correct == expected else "✗"
        print(f"{status} Output: '{output[:40]}...' -> {is_correct} (expected {expected})")
    
    print("-" * 60)
    print("Done!")
