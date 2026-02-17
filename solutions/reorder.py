#!/usr/bin/env python3
"""
Document Reordering Strategies for Small LLMs

Based on experimental findings:
- Gemma-2B: Recency bias (best at end, worst at beginning)
- Gemma-4B: Weak "Lost in Middle" (worst at position 50)
- Llama-3B: Flat/stable (no significant pattern)

These strategies reorder retrieved documents to place the most
relevant content in positions where the model pays most attention.
"""

from typing import List, Dict, Any, Literal

ModelType = Literal["gemma-2b", "gemma-4b", "llama-3b", "auto"]


def reorder_for_gemma_2b(documents: List[Dict[str, Any]], score_key: str = "score") -> List[Dict[str, Any]]:
    """
    Reorder documents for Gemma-2B's recency bias.

    Strategy: "Best-Last" - Place highest-scoring documents at the END
    of the context where Gemma-2B pays most attention.

    Args:
        documents: List of document dicts with scores (higher = more relevant)
        score_key: Key in dict containing relevance score

    Returns:
        Reordered documents with best content at the end

    Example:
        >>> docs = [{"text": "A", "score": 0.9}, {"text": "B", "score": 0.7}, {"text": "C", "score": 0.5}]
        >>> reordered = reorder_for_gemma_2b(docs)
        >>> # Result: [C, B, A] - best document "A" is now last
    """
    # Sort by score ascending so highest scores end up at the end
    sorted_docs = sorted(documents, key=lambda x: x.get(score_key, 0))
    return sorted_docs


def reorder_for_gemma_4b(documents: List[Dict[str, Any]], score_key: str = "score") -> List[Dict[str, Any]]:
    """
    Reorder documents for Gemma-4B's weak "Lost in Middle" pattern.

    Strategy: "Sides-First" - Place best documents at beginning AND end,
    lower-confidence documents in the middle.

    Args:
        documents: List of document dicts with scores (higher = more relevant)
        score_key: Key in dict containing relevance score

    Returns:
        Reordered documents with best content at edges

    Example:
        >>> docs = [{"text": "A", "score": 0.9}, {"text": "B", "score": 0.8},
        ...         {"text": "C", "score": 0.7}, {"text": "D", "score": 0.6}]
        >>> reordered = reorder_for_gemma_4b(docs)
        >>> # Result: [A, C, D, B] - best at start, second-best at end, rest in middle
    """
    if len(documents) <= 2:
        return documents

    # Sort by score descending
    sorted_docs = sorted(documents, key=lambda x: x.get(score_key, 0), reverse=True)

    # Distribute: best docs alternate between start and end
    result = [None] * len(sorted_docs)
    left, right = 0, len(sorted_docs) - 1

    for i, doc in enumerate(sorted_docs):
        if i % 2 == 0:
            result[left] = doc
            left += 1
        else:
            result[right] = doc
            right -= 1

    return result


def reorder_for_llama_3b(documents: List[Dict[str, Any]], score_key: str = "score") -> List[Dict[str, Any]]:
    """
    Reorder documents for Llama-3B.

    Strategy: Standard descending order (best first).
    Llama-3B showed flat/stable performance across positions,
    so standard relevance ordering works fine.

    Args:
        documents: List of document dicts with scores (higher = more relevant)
        score_key: Key in dict containing relevance score

    Returns:
        Documents sorted by score descending (standard RAG order)
    """
    return sorted(documents, key=lambda x: x.get(score_key, 0), reverse=True)


def reorder_documents(
    documents: List[Dict[str, Any]],
    model: ModelType = "auto",
    score_key: str = "score"
) -> List[Dict[str, Any]]:
    """
    Automatically reorder documents based on target model's attention pattern.

    Args:
        documents: List of document dicts with scores
        model: Target model ("gemma-2b", "gemma-4b", "llama-3b", or "auto")
        score_key: Key in dict containing relevance score

    Returns:
        Reordered documents optimized for the target model

    Example:
        >>> from solutions.reorder import reorder_documents
        >>>
        >>> # Your retrieved documents from vector DB
        >>> docs = [
        ...     {"text": "The capital of France is Paris.", "score": 0.95},
        ...     {"text": "France is in Europe.", "score": 0.82},
        ...     {"text": "Paris has the Eiffel Tower.", "score": 0.78},
        ... ]
        >>>
        >>> # Reorder for Gemma-2B (puts best doc last)
        >>> optimized = reorder_documents(docs, model="gemma-2b")
    """
    strategies = {
        "gemma-2b": reorder_for_gemma_2b,  # Best-last (recency bias)
        "gemma-4b": reorder_for_gemma_2b,  # Best-last (also has recency bias: Pos 100=97.2% > Pos 1=91.7%)
        "llama-3b": reorder_for_llama_3b,  # Standard order (stable across positions)
        "auto": reorder_for_gemma_2b,  # Default to recency bias handling
    }

    strategy = strategies.get(model.lower(), reorder_for_gemma_2b)
    return strategy(documents, score_key)


def reorder_with_gold_position(
    documents: List[Dict[str, Any]],
    gold_index: int,
    target_position: int,
) -> List[Dict[str, Any]]:
    """
    Move a specific document to a target position.

    Useful for testing or when you know exactly which document
    contains the answer and want to place it optimally.

    Args:
        documents: List of documents
        gold_index: Current index of the "gold" document
        target_position: Where to place the gold document (0-indexed)

    Returns:
        Reordered documents with gold at target position
    """
    if gold_index < 0 or gold_index >= len(documents):
        raise ValueError(f"gold_index {gold_index} out of range")
    if target_position < 0 or target_position >= len(documents):
        raise ValueError(f"target_position {target_position} out of range")

    docs = documents.copy()
    gold_doc = docs.pop(gold_index)
    docs.insert(target_position, gold_doc)
    return docs


# Convenience functions for common use cases

def best_last(documents: List[Dict[str, Any]], score_key: str = "score") -> List[Dict[str, Any]]:
    """Alias for reorder_for_gemma_2b - puts best documents at the end."""
    return reorder_for_gemma_2b(documents, score_key)


def sides_first(documents: List[Dict[str, Any]], score_key: str = "score") -> List[Dict[str, Any]]:
    """Alias for reorder_for_gemma_4b - puts best documents at edges."""
    return reorder_for_gemma_4b(documents, score_key)


def best_first(documents: List[Dict[str, Any]], score_key: str = "score") -> List[Dict[str, Any]]:
    """Standard descending order - puts best documents first."""
    return reorder_for_llama_3b(documents, score_key)


if __name__ == "__main__":
    # Demo
    sample_docs = [
        {"id": 1, "text": "The capital of Valdoria is Zenith City.", "score": 0.95},
        {"id": 2, "text": "Valdoria is located in the northern hemisphere.", "score": 0.72},
        {"id": 3, "text": "The population of Valdoria is 5 million.", "score": 0.68},
        {"id": 4, "text": "Valdoria's currency is the Valdorian Crown.", "score": 0.65},
        {"id": 5, "text": "Zenith City was founded in 1823.", "score": 0.60},
    ]

    print("Original order:")
    for doc in sample_docs:
        print(f"  [{doc['score']:.2f}] {doc['text'][:50]}...")

    print("\n--- Gemma-2B (Best-Last) ---")
    for doc in reorder_for_gemma_2b(sample_docs):
        print(f"  [{doc['score']:.2f}] {doc['text'][:50]}...")

    print("\n--- Gemma-4B (Sides-First) ---")
    for doc in reorder_for_gemma_4b(sample_docs):
        print(f"  [{doc['score']:.2f}] {doc['text'][:50]}...")

    print("\n--- Llama-3B (Best-First) ---")
    for doc in reorder_for_llama_3b(sample_docs):
        print(f"  [{doc['score']:.2f}] {doc['text'][:50]}...")
