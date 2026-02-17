"""Build context with gold document at specific position - V2 with hard distractors."""

import random
from typing import List, Dict


def build_context(
    qa_pair: Dict,
    generic_distractors: List[str],
    gold_position: int,
    total_docs: int = 50,
    seed: int = None,
) -> str:
    """
    Build a context string with the gold document at a specific position.

    Uses hard distractors (same-entity confusion) first, then fills with generic ones.

    Args:
        qa_pair: Dict with question, answer, gold_doc, and hard_distractors
        generic_distractors: List of generic distractor documents
        gold_position: Position for gold document (1-indexed)
        total_docs: Total number of documents in context
        seed: Random seed for reproducibility

    Returns:
        Formatted context string with all documents
    """
    if seed is not None:
        random.seed(seed)

    gold_doc = qa_pair["gold_doc"]
    hard_distractors = qa_pair.get("hard_distractors", [])

    # Calculate how many distractors we need (excluding gold doc position)
    num_distractors_needed = total_docs - 1

    # Use all hard distractors first, then fill with generic ones
    all_distractors = hard_distractors.copy()

    # Shuffle generic distractors and add enough to fill
    shuffled_generic = generic_distractors.copy()
    random.shuffle(shuffled_generic)

    remaining_needed = num_distractors_needed - len(all_distractors)
    if remaining_needed > 0:
        all_distractors.extend(shuffled_generic[:remaining_needed])

    # Shuffle all distractors
    random.shuffle(all_distractors)

    # Build document list with gold at specified position
    documents = []
    distractor_idx = 0

    for i in range(1, total_docs + 1):
        if i == gold_position:
            documents.append(gold_doc)
        else:
            documents.append(all_distractors[distractor_idx])
            distractor_idx += 1

    # Format as numbered documents
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(f"Document {i}: {doc}")

    return "\n\n".join(context_parts)


def build_prompt(context: str, question: str) -> str:
    """
    Build the full prompt with context and question.

    Args:
        context: The formatted document context
        question: The question to answer

    Returns:
        Complete prompt string
    """
    prompt = f"""Based on the following documents, answer the question. Give only the answer, no explanation.

{context}

Question: {question}
Answer:"""

    return prompt


if __name__ == "__main__":
    # Test the context builder
    import json
    from pathlib import Path

    data_dir = Path(__file__).parent.parent / "data"

    with open(data_dir / "qa_pairs.json") as f:
        qa_pairs = json.load(f)

    with open(data_dir / "distractors.json") as f:
        distractors = json.load(f)

    # Test with first QA pair
    qa = qa_pairs[0]
    context = build_context(qa, distractors, gold_position=25, total_docs=50, seed=42)
    prompt = build_prompt(context, qa["question"])

    print("=" * 60)
    print("TEST: Gold document at position 25 of 50")
    print("=" * 60)
    print(f"Question: {qa['question']}")
    print(f"Expected answer: {qa['answer']}")
    print(f"\nHard distractors included: {len(qa.get('hard_distractors', []))}")
    print(f"\nPrompt length: {len(prompt)} chars")
    print(f"\nFirst 3 documents:")
    for line in prompt.split("\n\n")[:3]:
        print(f"  {line[:100]}...")
    print(f"\nDocument 25 (gold):")
    lines = prompt.split("\n\n")
    print(f"  {lines[24][:100]}...")
