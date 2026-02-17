#!/usr/bin/env python3
"""
Lost in the Middle Experiment Runner - V2 (Harder)

Run with:
    python run_experiment.py --model gemma-2b --verbose --limit 5
    python run_experiment.py --dry-run
    python run_experiment.py  # Run all models
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MODELS,
    MODEL_CONFIG,
    TRIALS_PER_POSITION,
    MAX_NEW_TOKENS,
    TEMPERATURE,
)
from context_builder import build_context, build_prompt
from evaluator import check_answer
from model_runner import ModelRunner, DryRunModelRunner


def validate_data(qa_pairs: list, distractors: list, positions: list, total_docs: int):
    """Validate inputs before running experiment."""
    required_qa_fields = {"id", "question", "answer", "gold_doc"}
    for i, qa in enumerate(qa_pairs):
        missing = required_qa_fields - set(qa.keys())
        if missing:
            raise ValueError(f"QA pair {i} (id={qa.get('id')}) missing: {missing}")
    if not qa_pairs:
        raise ValueError("No QA pairs loaded")
    if not distractors:
        raise ValueError("No distractors loaded")
    for pos in positions:
        if not 1 <= pos <= total_docs:
            raise ValueError(f"Position {pos} must be 1..{total_docs}")
    num_distractors_needed = total_docs - 1
    max_hard = max(len(qa.get("hard_distractors", [])) for qa in qa_pairs)
    if len(distractors) + max_hard < num_distractors_needed:
        raise ValueError(
            f"Need {num_distractors_needed} distractors, have {len(distractors)} generic + up to {max_hard} hard"
        )


def load_data():
    """Load QA pairs and distractors."""
    data_dir = Path(__file__).parent.parent / "data"

    with open(data_dir / "qa_pairs.json") as f:
        qa_pairs = json.load(f)

    with open(data_dir / "distractors.json") as f:
        distractors = json.load(f)

    return qa_pairs, distractors


def run_experiment(
    model_name: str,
    model_path: str,
    qa_pairs: list,
    distractors: list,
    positions: list,
    total_docs: int,
    trials_per_position: int,
    dry_run: bool = False,
    verbose: bool = False,
    limit: int = None,
):
    """
    Run the experiment for a single model.

    Args:
        model_name: Name of the model
        model_path: Path to model weights
        qa_pairs: List of QA pairs
        distractors: List of distractor documents
        positions: List of positions to test
        total_docs: Total documents per context
        trials_per_position: Number of trials per position
        dry_run: If True, use mock model
        verbose: If True, print each response
        limit: If set, limit trials per position

    Returns:
        Dict with results
    """
    # Apply limit
    effective_trials = min(trials_per_position, len(qa_pairs))
    if limit:
        effective_trials = min(effective_trials, limit)

    # Initialize model
    if dry_run:
        runner = DryRunModelRunner(model_path)
    else:
        runner = ModelRunner(model_path)

    runner.load()

    results = {
        "model_name": model_name,
        "model_id": model_path,
        "config": {"positions": positions, "total_docs": total_docs},
        "positions": {},
        "raw_results": [],
    }

    for pos in positions:
        print(f"\n--- Position {pos}/{total_docs} ---")

        correct = 0
        total = 0

        qa_subset = qa_pairs[:effective_trials]
        pbar = tqdm(qa_subset, desc=f"Pos {pos}", leave=True)

        for i, qa in enumerate(pbar):
            # Build context with gold at this position
            seed = int(hashlib.sha256(f"{qa['id']}_{pos}".encode()).hexdigest(), 16) % (2**32)
            context = build_context(
                qa, distractors, gold_position=pos, total_docs=total_docs, seed=seed
            )
            prompt = build_prompt(context, qa["question"])

            # Generate response
            response, latency_ms = runner.generate(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )

            # Evaluate
            is_correct, extracted = check_answer(response, qa["answer"])

            if is_correct:
                correct += 1
            total += 1

            # Store result
            result = {
                "correct": is_correct,
                "response": response,
                "extracted": extracted,
                "gold_answer": qa["answer"],
                "latency_ms": latency_ms,
                "qa_id": qa["id"],
                "question": qa["question"],
                "position": pos,
            }
            results["raw_results"].append(result)

            # Verbose output
            if verbose:
                status = "✓" if is_correct else "✗"
                tqdm.write(f"\n{status} Q: {qa['question']}")
                tqdm.write(f"   Expected: {qa['answer']}")
                tqdm.write(
                    f"   Got: {response[:100]}{'...' if len(response) > 100 else ''}"
                )
                if not is_correct:
                    tqdm.write(f"   Extracted: {extracted}")

            pbar.set_postfix({"acc": f"{correct}/{total}"})

        accuracy = correct / total if total > 0 else 0
        results["positions"][str(pos)] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

        print(f"Position {pos}: {correct}/{total} = {accuracy:.1%}")

    runner.unload()

    return results


def main():
    parser = argparse.ArgumentParser(description="Lost in the Middle Experiment V2")
    parser.add_argument(
        "--model", type=str, choices=list(MODELS.keys()), help="Run specific model only"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without actual model inference"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print each response"
    )
    parser.add_argument("--limit", "-l", type=int, help="Limit trials per position")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (default: auto-generated with timestamp and model)",
    )

    args = parser.parse_args()

    # Load data
    qa_pairs, distractors = load_data()

    # Determine which models to run
    models_to_run = {args.model: MODELS[args.model]} if args.model else MODELS

    # Calculate effective trials (capped by available QA pairs)
    requested_trials = args.limit or TRIALS_PER_POSITION
    effective_trials = min(requested_trials, len(qa_pairs))

    print(f"Loaded {len(qa_pairs)} QA pairs (with hard distractors)")
    print(f"Loaded {len(distractors)} generic distractors")

    if effective_trials < requested_trials:
        print(f"WARNING: Requested {requested_trials} trials but only {len(qa_pairs)} QA pairs available.")
        print(f"         Capping trials_per_position to {effective_trials}.")

    print(f"Trials per position: {effective_trials}")
    if args.verbose:
        print("Verbose mode: ON")

    # Generate output filename with timestamp and model name(s)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = Path(__file__).parent.parent / args.output
    else:
        model_names = "_".join(models_to_run.keys())
        output_path = (
            Path(__file__).parent.parent
            / f"results/results_{model_names}_{timestamp}.json"
        )

    # Run experiments
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "trials_per_position": effective_trials,  # Actual trials run, not requested
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
        },
        "models": {},
    }

    for model_name, model_path in models_to_run.items():
        cfg = MODEL_CONFIG[model_name]
        positions = cfg["positions"]
        total_docs = cfg["total_docs"]

        validate_data(qa_pairs, distractors, positions, total_docs)

        print(f"\n{'='*60}")
        print(f"Running experiment: {model_name} ({total_docs} docs)")
        print(f"Model: {model_path}")
        print(f"Positions: {positions}")
        print(f"{'='*60}")

        results = run_experiment(
            model_name=model_name,
            model_path=model_path,
            qa_pairs=qa_pairs,
            distractors=distractors,
            positions=positions,
            total_docs=total_docs,
            trials_per_position=TRIALS_PER_POSITION,
            dry_run=args.dry_run,
            verbose=args.verbose,
            limit=args.limit,
        )

        all_results["models"][model_name] = results

    # Backward compat: top-level config from first model (for visualize.py)
    first_model = next(iter(all_results["models"]))
    all_results["config"].update(all_results["models"][first_model]["config"])

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary (each model may have different positions)
    print(f"\n{'='*70}")
    print("SUMMARY: Accuracy by Position")
    print(f"{'='*70}")

    for model_name, results in all_results["models"].items():
        positions = results["config"]["positions"]
        header = "Model".ljust(15) + " | " + " | ".join([f"Pos {p:>2}" for p in positions])
        print(header)
        print("-" * len(header))
        row = model_name.ljust(15) + " | "
        row += " | ".join(
            [
                f"{results['positions'][str(p)]['accuracy']*100:>5.1f}%"
                for p in positions
            ]
        )
        print(row)
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()
