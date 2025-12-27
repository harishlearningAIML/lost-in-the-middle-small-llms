"""
Run Experiment - Main script for Lost in the Middle testing

Usage:
    python run_experiment.py                    # Run all models
    python run_experiment.py --model gemma-2b   # Run single model
    python run_experiment.py --dry-run          # Test without model
    python run_experiment.py --verbose          # Show detailed output per test
    python run_experiment.py -v --dry-run       # Combine flags
"""

import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm

from config import MODELS, POSITIONS, TOTAL_DOCS, TRIALS_PER_POSITION, MAX_NEW_TOKENS, TEMPERATURE
from context_builder import load_qa_pairs, load_distractors, build_prompt
from evaluator import check_answer


def run_single_test(
    model_runner,
    question: str,
    gold_doc: str,
    gold_answer: str,
    answer_variants: List[str],
    distractors: List[str],
    position: int,
    seed: int,
    verbose: bool = False
) -> Dict:
    """Run a single test and return results"""
    
    # Build prompt
    prompt = build_prompt(
        question=question,
        gold_doc=gold_doc,
        distractors=distractors,
        gold_position=position,
        total_docs=TOTAL_DOCS,
        seed=seed
    )
    
    # Generate response
    start = time.time()
    if model_runner is None:
        # Dry run mode
        response = "[DRY RUN - no model loaded]"
    else:
        response = model_runner.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE
        )
    elapsed = time.time() - start
    
    # Evaluate
    is_correct, extracted = check_answer(response, gold_answer, answer_variants)

    # Verbose output
    if verbose:
        print(f"\n{'─'*50}")
        print(f"Question: {question}")
        print(f"Gold position: {position}/{TOTAL_DOCS}")
        print(f"Expected answer: {gold_answer}")
        print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        print(f"Extracted: {extracted}")
        print(f"Correct: {'✓' if is_correct else '✗'}")
        print(f"Latency: {int(elapsed * 1000)}ms")

    return {
        "correct": is_correct,
        "response": response,
        "extracted": extracted,
        "gold_answer": gold_answer,
        "latency_ms": int(elapsed * 1000)
    }


def run_experiment_for_model(
    model_name: str,
    model_id: str,
    qa_pairs: List[Dict],
    distractors: List[str],
    dry_run: bool = False,
    verbose: bool = False
) -> Dict:
    """Run full experiment for a single model"""
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {model_name}")
    print(f"Model: {model_id}")
    print(f"{'='*60}")
    
    # Load model
    model_runner = None
    if not dry_run:
        from model_runner import ModelRunner
        model_runner = ModelRunner(model_id)
        model_runner.load()
    
    results = {
        "model_name": model_name,
        "model_id": model_id,
        "positions": {},
        "raw_results": []
    }
    
    # Run tests for each position
    for position in POSITIONS:
        print(f"\n--- Position {position}/{TOTAL_DOCS} ---")
        
        position_results = []
        num_trials = min(TRIALS_PER_POSITION, len(qa_pairs))
        
        for i in tqdm(range(num_trials), desc=f"Pos {position}"):
            qa = qa_pairs[i]
            
            result = run_single_test(
                model_runner=model_runner,
                question=qa["question"],
                gold_doc=qa["gold_doc"],
                gold_answer=qa["answer"],
                answer_variants=qa.get("answer_variants", []),
                distractors=distractors,
                position=position,
                seed=position * 1000 + i,  # Reproducible but different per position
                verbose=verbose
            )
            
            result["qa_id"] = qa["id"]
            result["question"] = qa["question"]
            result["position"] = position
            
            position_results.append(result)
            results["raw_results"].append(result)
        
        # Calculate accuracy for this position
        correct = sum(1 for r in position_results if r["correct"])
        accuracy = correct / len(position_results) if position_results else 0
        
        results["positions"][position] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(position_results)
        }
        
        print(f"Position {position}: {correct}/{len(position_results)} = {accuracy:.1%}")
    
    # Unload model
    if model_runner is not None:
        model_runner.unload()
    
    return results


def run_all_experiments(
    models: Dict[str, str] = None,
    dry_run: bool = False,
    verbose: bool = False
) -> Dict:
    """Run experiments for all models"""
    
    if models is None:
        models = MODELS
    
    # Load data
    qa_pairs = load_qa_pairs()
    distractors = load_distractors()
    
    print(f"Loaded {len(qa_pairs)} QA pairs")
    print(f"Loaded {len(distractors)} distractors")
    print(f"Testing positions: {POSITIONS}")
    print(f"Total docs per context: {TOTAL_DOCS}")
    print(f"Trials per position: {min(TRIALS_PER_POSITION, len(qa_pairs))}")
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "positions": POSITIONS,
            "total_docs": TOTAL_DOCS,
            "trials_per_position": min(TRIALS_PER_POSITION, len(qa_pairs)),
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE
        },
        "models": {}
    }
    
    for model_name, model_id in models.items():
        try:
            results = run_experiment_for_model(
                model_name=model_name,
                model_id=model_id,
                qa_pairs=qa_pairs,
                distractors=distractors,
                dry_run=dry_run,
                verbose=verbose
            )
            all_results["models"][model_name] = results
        except Exception as e:
            print(f"ERROR running {model_name}: {e}")
            all_results["models"][model_name] = {"error": str(e)}
    
    return all_results


def print_summary(results: Dict):
    """Print a summary table of results"""
    
    print("\n" + "="*70)
    print("SUMMARY: Accuracy by Position")
    print("="*70)
    
    # Header
    positions = results["config"]["positions"]
    header = f"{'Model':<15} | " + " | ".join([f"Pos {p:>2}" for p in positions])
    print(header)
    print("-" * len(header))
    
    # Data rows
    for model_name, model_data in results["models"].items():
        if "error" in model_data:
            print(f"{model_name:<15} | ERROR: {model_data['error'][:40]}")
            continue
            
        row = f"{model_name:<15} | "
        accuracies = []
        for pos in positions:
            acc = model_data["positions"][pos]["accuracy"]
            accuracies.append(f"{acc:>5.1%}")
        row += " | ".join(accuracies)
        print(row)
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Run Lost in the Middle experiment")
    parser.add_argument("--model", type=str, help="Run single model (e.g., gemma-2b)")
    parser.add_argument("--dry-run", action="store_true", help="Test without loading models")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output for each test")
    parser.add_argument("--output", type=str, help="Output file (default: auto-generated with timestamp and model)")
    args = parser.parse_args()

    # Create results directory
    Path("results").mkdir(exist_ok=True)

    # Select models
    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(MODELS.keys())}")
            return
        models = {args.model: MODELS[args.model]}
    else:
        models = MODELS

    # Generate output filename with timestamp and model name(s)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_file = args.output
    else:
        model_names = "_".join(models.keys())
        output_file = f"results/results_{model_names}_{timestamp}.json"

    # Run experiments
    results = run_all_experiments(models=models, dry_run=args.dry_run, verbose=args.verbose)
    
    # Print summary
    print_summary(results)
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
