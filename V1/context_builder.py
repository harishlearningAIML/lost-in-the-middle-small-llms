"""
Context Builder - Creates prompts with gold document at specified position
"""

import json
import random
from pathlib import Path


def load_qa_pairs(path: str = "data/qa_pairs.json") -> list:
    """Load question-answer pairs"""
    with open(path, "r") as f:
        return json.load(f)


def load_distractors(path: str = "data/distractors.json") -> list:
    """Load distractor documents"""
    with open(path, "r") as f:
        return json.load(f)


def build_context(
    gold_doc: str,
    distractors: list,
    gold_position: int,
    total_docs: int = 20,
    seed: int = None
) -> str:
    """
    Build context with gold document at specified position.
    
    Args:
        gold_doc: The document containing the answer
        distractors: List of distractor documents
        gold_position: Position for gold doc (1-indexed, 1 to total_docs)
        total_docs: Total number of documents in context
        seed: Random seed for reproducibility
    
    Returns:
        Formatted context string with numbered documents
    """
    if seed is not None:
        random.seed(seed)
    
    # Select distractors (need total_docs - 1)
    num_distractors = total_docs - 1
    if len(distractors) < num_distractors:
        # Sample with replacement if not enough distractors
        selected = random.choices(distractors, k=num_distractors)
    else:
        selected = random.sample(distractors, num_distractors)
    
    # Insert gold doc at specified position (convert to 0-indexed)
    docs = selected[:gold_position - 1] + [gold_doc] + selected[gold_position - 1:]
    
    # Format as numbered documents
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"Document {i}: {doc}")
    
    return "\n\n".join(context_parts)


def build_prompt(
    question: str,
    gold_doc: str,
    distractors: list,
    gold_position: int,
    total_docs: int = 20,
    seed: int = None
) -> str:
    """
    Build complete prompt for the model.
    
    Args:
        question: The question to answer
        gold_doc: Document containing the answer
        distractors: List of distractor documents
        gold_position: Where to place the gold document
        total_docs: Total documents in context
        seed: Random seed
    
    Returns:
        Complete prompt string
    """
    context = build_context(gold_doc, distractors, gold_position, total_docs, seed)
    
    prompt = f"""Based on the following documents, answer the question. Give a brief, direct answer.

{context}

Question: {question}
Answer:"""
    
    return prompt


# Quick test
if __name__ == "__main__":
    qa_pairs = load_qa_pairs()
    distractors = load_distractors()
    
    # Test with first QA pair
    qa = qa_pairs[0]
    
    print("=" * 60)
    print(f"Question: {qa['question']}")
    print(f"Expected Answer: {qa['answer']}")
    print("=" * 60)
    
    # Build prompt with gold at position 10
    prompt = build_prompt(
        question=qa["question"],
        gold_doc=qa["gold_doc"],
        distractors=distractors,
        gold_position=10,
        total_docs=20,
        seed=42
    )
    
    print("\nGenerated Prompt (truncated):")
    print("-" * 60)
    # Show first 500 and last 500 chars
    if len(prompt) > 1200:
        print(prompt[:600])
        print("\n... [middle truncated] ...\n")
        print(prompt[-600:])
    else:
        print(prompt)
