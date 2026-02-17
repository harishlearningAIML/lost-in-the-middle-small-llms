#!/usr/bin/env python3
"""
Comparison Experiment: Baseline vs Solution

This script compares:
1. BASELINE: Standard RAG order (best documents first)
2. SOLUTION: Reordered documents based on model bias

Results are saved to solutions/results/

Usage:
    python solutions/run_comparison.py
    python solutions/run_comparison.py --model gemma-2b
    python solutions/run_comparison.py --trials 10
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from solutions import reorder_documents


def load_experiment_results():
    """Load the actual experiment results."""
    results_dir = Path(__file__).parent.parent / "results"

    return {
        "gemma-2b": json.load(open(results_dir / "results_gemma-2b_20260211_091248.json")),
        "gemma-4b": json.load(open(results_dir / "results_gemma-4b_20260211_094409.json")),
        "llama-3b": json.load(open(results_dir / "results_llama-3b_20260211_105815.json")),
    }


def simulate_rag_retrieval(n_docs: int = 10) -> list:
    """
    Simulate RAG retrieval results.

    In real usage, these would come from your vector DB.
    The "gold" document (with the answer) has the highest score.
    """
    docs = []
    for i in range(n_docs):
        score = 0.95 - (i * 0.05)  # Descending scores
        docs.append({
            "id": i,
            "text": f"Document {i}" if i > 0 else "GOLD: The answer is here.",
            "score": max(score, 0.1),
            "is_gold": i == 0
        })
    return docs


def get_gold_position(docs: list) -> int:
    """Find the position of the gold document (1-indexed)."""
    for i, doc in enumerate(docs):
        if doc.get("is_gold", False):
            return i + 1
    return -1


def estimate_accuracy(model: str, position: int, total_docs: int, results: dict) -> float:
    """
    Estimate accuracy based on experimental results.

    Interpolates between tested positions.
    """
    model_key = list(results[model]["models"].keys())[0]
    positions = results[model]["config"]["positions"]
    pos_data = results[model]["models"][model_key]["positions"]

    # Normalize position to percentage of context
    pct = (position / total_docs) * 100

    # Find closest tested positions
    tested_pcts = [(p / results[model]["config"]["total_docs"]) * 100 for p in positions]

    # Find bracketing positions
    lower_idx = 0
    upper_idx = len(positions) - 1

    for i, test_pct in enumerate(tested_pcts):
        if test_pct <= pct:
            lower_idx = i
        if test_pct >= pct:
            upper_idx = i
            break

    # Get accuracies
    lower_acc = pos_data[str(positions[lower_idx])]["accuracy"]
    upper_acc = pos_data[str(positions[upper_idx])]["accuracy"]

    # Interpolate
    if lower_idx == upper_idx:
        return lower_acc * 100

    lower_pct = tested_pcts[lower_idx]
    upper_pct = tested_pcts[upper_idx]

    if upper_pct == lower_pct:
        return lower_acc * 100

    ratio = (pct - lower_pct) / (upper_pct - lower_pct)
    estimated = lower_acc + ratio * (upper_acc - lower_acc)

    return estimated * 100


def run_comparison(model: str, n_docs: int, n_trials: int, results: dict) -> dict:
    """
    Run comparison between baseline and solution.

    Returns dict with comparison results.
    """
    print(f"\n{'='*60}")
    print(f"MODEL: {model.upper()}")
    print(f"Documents: {n_docs}, Trials: {n_trials}")
    print(f"{'='*60}")

    baseline_accuracies = []
    solution_accuracies = []

    total_docs = results[model]["config"]["total_docs"]

    for trial in range(n_trials):
        # Simulate retrieval (gold doc has highest score)
        docs = simulate_rag_retrieval(n_docs)

        # BASELINE: Standard order (best first)
        baseline_docs = sorted(docs, key=lambda x: x["score"], reverse=True)
        baseline_pos = get_gold_position(baseline_docs)
        baseline_acc = estimate_accuracy(model, baseline_pos, n_docs, results)
        baseline_accuracies.append(baseline_acc)

        # SOLUTION: Reordered for model
        solution_docs = reorder_documents(docs, model=model)
        solution_pos = get_gold_position(solution_docs)
        solution_acc = estimate_accuracy(model, solution_pos, n_docs, results)
        solution_accuracies.append(solution_acc)

        if trial == 0:  # Show first trial details
            print(f"\nTrial 1 Details:")
            print(f"  Baseline: Gold at position {baseline_pos}/{n_docs} → {baseline_acc:.1f}%")
            print(f"  Solution: Gold at position {solution_pos}/{n_docs} → {solution_acc:.1f}%")

    # Calculate averages
    avg_baseline = sum(baseline_accuracies) / len(baseline_accuracies)
    avg_solution = sum(solution_accuracies) / len(solution_accuracies)
    improvement = avg_solution - avg_baseline

    print(f"\nResults ({n_trials} trials):")
    print(f"  BASELINE (best-first): {avg_baseline:.1f}% avg accuracy")
    print(f"  SOLUTION (reordered):  {avg_solution:.1f}% avg accuracy")
    print(f"  IMPROVEMENT:           {improvement:+.1f}%")

    return {
        "model": model,
        "n_docs": n_docs,
        "n_trials": n_trials,
        "baseline": {
            "strategy": "best-first",
            "avg_accuracy": avg_baseline,
            "gold_position": 1,  # Always first in baseline
        },
        "solution": {
            "strategy": "best-last" if model in ("gemma-2b", "gemma-4b") else "best-first",
            "avg_accuracy": avg_solution,
            "gold_position": n_docs if model in ("gemma-2b", "gemma-4b") else 1,
        },
        "improvement": improvement,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs solution")
    parser.add_argument("--model", type=str, choices=["gemma-2b", "gemma-4b", "llama-3b", "all"],
                        default="all", help="Model to test")
    parser.add_argument("--docs", type=int, default=10, help="Number of documents")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--output", type=str, help="Output file path")

    args = parser.parse_args()

    print("=" * 60)
    print("COMPARISON EXPERIMENT: Baseline vs Solution")
    print("=" * 60)

    # Load experimental results
    print("\nLoading experimental results...")
    results = load_experiment_results()

    # Determine models to test
    models = ["gemma-2b", "gemma-4b", "llama-3b"] if args.model == "all" else [args.model]

    # Run comparisons
    all_comparisons = {}

    for model in models:
        comparison = run_comparison(model, args.docs, args.trials, results)
        all_comparisons[model] = comparison

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<12} | {'Baseline':<10} | {'Solution':<10} | {'Improvement':<12}")
    print("-" * 50)

    for model, comp in all_comparisons.items():
        print(f"{model:<12} | {comp['baseline']['avg_accuracy']:>8.1f}% | {comp['solution']['avg_accuracy']:>8.1f}% | {comp['improvement']:>+10.1f}%")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = output_dir / args.output
    else:
        output_path = output_dir / f"comparison_{timestamp}.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_docs": args.docs,
            "n_trials": args.trials,
        },
        "comparisons": all_comparisons,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    for model, comp in all_comparisons.items():
        if comp["improvement"] > 5:
            print(f"\n{model}: USE SOLUTION")
            print(f"  Strategy: {comp['solution']['strategy']}")
            print(f"  Expected gain: {comp['improvement']:+.1f}%")
        elif comp["improvement"] > 0:
            print(f"\n{model}: OPTIONAL (small gain)")
            print(f"  Strategy: {comp['solution']['strategy']}")
            print(f"  Expected gain: {comp['improvement']:+.1f}%")
        else:
            print(f"\n{model}: NOT NEEDED")
            print(f"  Standard RAG order works fine")


if __name__ == "__main__":
    main()
