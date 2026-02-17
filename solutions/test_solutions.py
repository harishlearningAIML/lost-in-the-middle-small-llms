#!/usr/bin/env python3
"""
Test script for the solutions package.

Run with:
    python -m solutions.test_solutions
    # or
    python solutions/test_solutions.py
"""

import sys
from pathlib import Path

# Add parent to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from solutions import (
    reorder_documents,
    reorder_for_gemma_2b,
    reorder_for_gemma_4b,
    reorder_for_llama_3b,
    build_rag_prompt,
    create_pipeline,
)


def test_reordering():
    """Test document reordering strategies."""
    print("=" * 50)
    print("TEST: Document Reordering")
    print("=" * 50)

    docs = [
        {"id": "A", "text": "Best doc", "score": 0.9},
        {"id": "B", "text": "Second", "score": 0.7},
        {"id": "C", "text": "Third", "score": 0.5},
        {"id": "D", "text": "Fourth", "score": 0.3},
    ]

    # Test Gemma-2B (best-last)
    result = reorder_for_gemma_2b(docs)
    order = [d["id"] for d in result]
    expected = ["D", "C", "B", "A"]  # Ascending by score
    assert order == expected, f"Gemma-2B failed: {order} != {expected}"
    print(f"  Gemma-2B (best-last): {order} == {expected}")

    # Test Gemma-4B (sides-first)
    result = reorder_for_gemma_4b(docs)
    order = [d["id"] for d in result]
    # Best at edges: A at start, B at end, C and D in middle
    assert order[0] == "A", f"Gemma-4B should have best first: {order}"
    assert order[-1] == "B", f"Gemma-4B should have 2nd best last: {order}"
    print(f"  Gemma-4B (sides-first): {order} (best at edges)")

    # Test Llama-3B (best-first, standard)
    result = reorder_for_llama_3b(docs)
    order = [d["id"] for d in result]
    expected = ["A", "B", "C", "D"]  # Descending by score
    assert order == expected, f"Llama-3B failed: {order} != {expected}"
    print(f"  Llama-3B (best-first): {order} == {expected}")

    # Test auto-dispatch
    result = reorder_documents(docs, model="gemma-2b")
    order = [d["id"] for d in result]
    assert order == ["D", "C", "B", "A"], f"Auto dispatch failed: {order}"
    print(f"  Auto dispatch (gemma-2b): {order}")

    print("  All reordering tests passed!")


def test_prompts():
    """Test prompt generation."""
    print("\n" + "=" * 50)
    print("TEST: Prompt Generation")
    print("=" * 50)

    docs = [{"text": "Paris is the capital of France."}]
    query = "What is the capital of France?"

    prompt = build_rag_prompt(query, docs)

    # Check key elements
    assert "Paris is the capital" in prompt, "Missing document content"
    assert query in prompt, "Missing query"
    assert "Reminder" in prompt, "Missing attention refresher"

    print(f"  Prompt length: {len(prompt)} chars")
    print(f"  Contains document: Yes")
    print(f"  Contains query: Yes")
    print(f"  Contains reminder: Yes")
    print("  All prompt tests passed!")


def test_pipeline():
    """Test the RAG pipeline."""
    print("\n" + "=" * 50)
    print("TEST: RAG Pipeline")
    print("=" * 50)

    docs = [
        {"id": 1, "text": "The answer is 42.", "score": 0.95},
        {"id": 2, "text": "Unrelated info.", "score": 0.50},
        {"id": 3, "text": "More noise.", "score": 0.30},
    ]

    pipeline = create_pipeline("gemma-2b")
    result = pipeline.process_with_details("What is the answer?", docs)

    # Check reordering happened (best should be last for gemma-2b)
    assert result["after_reorder"][-1] == 1, "Best doc should be last"

    print(f"  Original order: {result['original_order']}")
    print(f"  After reorder: {result['after_reorder']}")
    print(f"  Config: {result['config']['model']}")
    print("  All pipeline tests passed!")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 50)
    print("TEST: Edge Cases")
    print("=" * 50)

    # Empty documents
    result = reorder_for_gemma_2b([])
    assert result == [], "Empty list should return empty"
    print("  Empty list: OK")

    # Single document
    single = [{"text": "Only one", "score": 1.0}]
    result = reorder_for_gemma_2b(single)
    assert len(result) == 1, "Single doc should stay single"
    print("  Single doc: OK")

    # Two documents
    two = [{"id": "A", "score": 0.9}, {"id": "B", "score": 0.5}]
    result = reorder_for_gemma_4b(two)
    assert len(result) == 2, "Two docs should stay two"
    print("  Two docs: OK")

    # Missing score key (should use default 0)
    no_score = [{"text": "No score"}, {"text": "Also no score"}]
    result = reorder_for_gemma_2b(no_score)
    assert len(result) == 2, "Docs without score should still work"
    print("  Missing score: OK")

    print("  All edge case tests passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("SOLUTIONS PACKAGE TEST SUITE")
    print("=" * 50)

    try:
        test_reordering()
        test_prompts()
        test_pipeline()
        test_edge_cases()
    except AssertionError as e:
        print(f"\n  FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    main()
