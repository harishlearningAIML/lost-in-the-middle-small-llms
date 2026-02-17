"""
Quick test with a single model - use this to verify setup before full experiment
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from context_builder import load_qa_pairs, load_distractors, build_prompt
from evaluator import check_answer


# Quick test config - modify as needed
MODEL_PATH = "/Volumes/T9/models/google/gemma-2-2b-it"  # Change to your model
TEST_POSITIONS = [1, 10, 20]  # Just test edges and middle
NUM_QUESTIONS = 3  # Quick test


def main():
    print("=" * 60)
    print("Quick Test: Lost in the Middle")
    print("=" * 60)
    
    # Load data
    qa_pairs = load_qa_pairs()[:NUM_QUESTIONS]
    distractors = load_distractors()
    
    print(f"Testing with {len(qa_pairs)} questions")
    print(f"Positions: {TEST_POSITIONS}")
    print(f"Model: {MODEL_PATH}")
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded on {model.device}")
    
    # Run tests
    results = {pos: {"correct": 0, "total": 0} for pos in TEST_POSITIONS}
    
    for qa in qa_pairs:
        print(f"\n{'─' * 60}")
        print(f"Q: {qa['question']}")
        print(f"Expected: {qa['answer']}")
        
        for position in TEST_POSITIONS:
            # Build prompt
            prompt = build_prompt(
                question=qa["question"],
                gold_doc=qa["gold_doc"],
                distractors=distractors,
                gold_position=position,
                total_docs=20,
                seed=42
            )
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Evaluate
            is_correct, extracted = check_answer(answer, qa["answer"], qa.get("answer_variants"))
            
            results[position]["total"] += 1
            if is_correct:
                results[position]["correct"] += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  Position {position:2d}: {status} → '{answer[:50]}...'")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    
    for pos in TEST_POSITIONS:
        acc = results[pos]["correct"] / results[pos]["total"] * 100
        print(f"Position {pos:2d}: {acc:.0f}% ({results[pos]['correct']}/{results[pos]['total']})")
    
    print("\nIf you see the U-shape (high→low→high), the effect is present!")
    print("Run the full experiment with: python run_experiment.py")


if __name__ == "__main__":
    main()
