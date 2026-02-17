#!/usr/bin/env python3
"""
Deep validation of specific results to ensure evaluator correctness.
Checks specific error cases and edge cases.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluator import check_answer


def validate_specific_errors():
    """Check specific incorrect answers from the results to ensure evaluator is working."""

    print("="*80)
    print("DEEP VALIDATION: Checking Specific Error Cases")
    print("="*80)

    # Load Gemma-2B results
    with open("results/results_gemma-2b_20251226_162353.json") as f:
        gemma2b_data = json.load(f)

    raw_results = gemma2b_data["models"]["gemma-2b"]["raw_results"]

    # Find all incorrect answers
    incorrect = [r for r in raw_results if not r["correct"]]

    print(f"\nFound {len(incorrect)} incorrect answers in Gemma-2B results")
    print(f"Total results: {len(raw_results)}")
    print(f"Error rate: {len(incorrect)/len(raw_results)*100:.1f}%\n")

    print("Checking each error case:")
    print("-" * 80)

    errors_validated = 0
    errors_failed = 0

    for err in incorrect:
        qa_id = err["qa_id"]
        position = err["position"]
        response = err["response"]
        extracted = err["extracted"]
        gold_answer = err["gold_answer"]

        # Re-evaluate
        is_correct, _ = check_answer(response, gold_answer)

        if is_correct == False:  # Should still be incorrect
            errors_validated += 1
            print(f"✓ {qa_id} pos {position:3d}: '{extracted}' != '{gold_answer}' (correctly marked wrong)")
        else:
            errors_failed += 1
            print(f"❌ {qa_id} pos {position:3d}: '{extracted}' vs '{gold_answer}' (SHOULD BE WRONG BUT EVALUATOR SAYS CORRECT!)")

    print(f"\n{'-' * 80}")
    print(f"Errors validated: {errors_validated}/{len(incorrect)}")

    if errors_failed > 0:
        print(f"❌ VALIDATION FAILED: {errors_failed} incorrect answers are being evaluated as correct!")
        return False
    else:
        print(f"✓ All error cases validated - evaluator is working correctly")
        return True


def validate_correct_answers():
    """Sample some correct answers to ensure evaluator isn't too lenient."""

    print("\n" + "="*80)
    print("DEEP VALIDATION: Checking Correct Answers")
    print("="*80)

    # Load Gemma-2B results
    with open("results/results_gemma-2b_20251226_162353.json") as f:
        gemma2b_data = json.load(f)

    raw_results = gemma2b_data["models"]["gemma-2b"]["raw_results"]

    # Find all correct answers
    correct = [r for r in raw_results if r["correct"]]

    print(f"\nFound {len(correct)} correct answers in Gemma-2B results")
    print(f"Sampling 20 to verify:\n")

    import random
    random.seed(42)
    sample = random.sample(correct, min(20, len(correct)))

    validated = 0
    failed = 0

    for res in sample:
        qa_id = res["qa_id"]
        position = res["position"]
        response = res["response"]
        extracted = res["extracted"]
        gold_answer = res["gold_answer"]

        # Re-evaluate
        is_correct, _ = check_answer(response, gold_answer)

        if is_correct == True:  # Should still be correct
            validated += 1
            print(f"✓ {qa_id} pos {position:3d}: '{extracted}' ≈ '{gold_answer}'")
        else:
            failed += 1
            print(f"❌ {qa_id} pos {position:3d}: '{extracted}' vs '{gold_answer}' (SHOULD BE CORRECT BUT EVALUATOR SAYS WRONG!)")

    print(f"\n{'-' * 80}")
    print(f"Correct answers validated: {validated}/{len(sample)}")

    if failed > 0:
        print(f"❌ VALIDATION FAILED: {failed} correct answers are being evaluated as incorrect!")
        return False
    else:
        print(f"✓ All sampled correct answers validated")
        return True


def check_position_placement():
    """Verify that gold documents are actually placed at the specified positions."""

    print("\n" + "="*80)
    print("DEEP VALIDATION: Gold Document Position Placement")
    print("="*80)

    print("\nVerifying position placement using context_builder function...")

    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from context_builder import build_context

    # Load data
    with open("data/qa_pairs.json") as f:
        qa_pairs = json.load(f)

    with open("data/distractors.json") as f:
        distractors = json.load(f)

    # Test a few positions
    test_cases = [
        (qa_pairs[0], 1, 100),
        (qa_pairs[0], 50, 100),
        (qa_pairs[0], 100, 100),
        (qa_pairs[5], 25, 100),
        (qa_pairs[10], 75, 100),
    ]

    print("\nTesting position placement for sample cases:")
    print("-" * 80)

    all_correct = True

    for qa_pair, target_position, total_docs in test_cases:
        qa_id = qa_pair["id"]
        gold_doc = qa_pair["gold_doc"]

        # Build context
        context = build_context(qa_pair, distractors, target_position, total_docs=total_docs, seed=42)

        # Find where gold doc actually is
        # Context format is "Document 1: ...\n\nDocument 2: ...\n\n..."
        lines = context.split("\n\n")

        gold_found_at = None
        for line in lines:
            if line.startswith("Document "):
                # Extract doc number and content
                parts = line.split(": ", 1)
                if len(parts) == 2:
                    doc_num_str = parts[0].replace("Document ", "")
                    doc_content = parts[1]

                    # Check if this is the gold doc
                    if gold_doc in doc_content:
                        gold_found_at = int(doc_num_str)
                        break

        if gold_found_at == target_position:
            print(f"✓ {qa_id} at position {target_position}: Gold doc correctly placed")
        else:
            print(f"❌ {qa_id} at position {target_position}: Gold doc found at position {gold_found_at}!")
            all_correct = False

    print(f"\n{'-' * 80}")
    if all_correct:
        print("✓ All position placements verified")
        return True
    else:
        print("❌ Position placement errors detected!")
        return False


def check_determinism():
    """Verify that building the same context twice gives the same result."""

    print("\n" + "="*80)
    print("DEEP VALIDATION: Determinism Check")
    print("="*80)

    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from context_builder import build_context

    # Load data
    with open("data/qa_pairs.json") as f:
        qa_pairs = json.load(f)

    with open("data/distractors.json") as f:
        distractors = json.load(f)

    qa_pair = qa_pairs[0]
    position = 50

    # Build twice with same seed
    context1 = build_context(qa_pair, distractors, position, total_docs=100, seed=42)
    context2 = build_context(qa_pair, distractors, position, total_docs=100, seed=42)

    if context1 == context2:
        print("✓ Context building is deterministic (same input + seed -> same output)")
        return True
    else:
        print("❌ Context building is NOT deterministic!")
        print(f"  Length 1: {len(context1)}")
        print(f"  Length 2: {len(context2)}")
        return False


def validate_qa_coverage():
    """Check that all 30 QA pairs are tested at each position."""

    print("\n" + "="*80)
    print("DEEP VALIDATION: QA Pair Coverage")
    print("="*80)

    with open("results/results_gemma-2b_20251226_162353.json") as f:
        data = json.load(f)

    raw_results = data["models"]["gemma-2b"]["raw_results"]
    positions = data["config"]["positions"]

    print(f"\nChecking that all 30 QA pairs are tested at each position:")
    print("-" * 80)

    coverage_ok = True

    for pos in positions:
        qa_ids_at_pos = [r["qa_id"] for r in raw_results if r["position"] == pos]
        unique_qa_ids = set(qa_ids_at_pos)

        if len(unique_qa_ids) != 30:
            print(f"❌ Position {pos}: Only {len(unique_qa_ids)} unique QA pairs (expected 30)")
            coverage_ok = False
        else:
            print(f"✓ Position {pos}: All 30 QA pairs tested")

    print(f"\n{'-' * 80}")
    if coverage_ok:
        print("✓ All positions have complete QA pair coverage")
        return True
    else:
        print("❌ Some positions are missing QA pairs!")
        return False


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE DEEP VALIDATION")
    print("="*80 + "\n")

    checks = [
        ("QA Pair Coverage", validate_qa_coverage),
        ("Incorrect Answer Cases", validate_specific_errors),
        ("Correct Answer Sampling", validate_correct_answers),
        ("Gold Document Placement", check_position_placement),
        ("Determinism", check_determinism),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ {name} check failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Final summary
    print("\n" + "="*80)
    print("FINAL VALIDATION SUMMARY")
    print("="*80 + "\n")

    for name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL VALIDATION CHECKS PASSED")
        print("="*80)
        print("\nThe results are VALID and TRUSTWORTHY.")
        return 0
    else:
        print("❌ SOME VALIDATION CHECKS FAILED")
        print("="*80)
        print("\nThere are issues with the results that need investigation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
