#!/usr/bin/env python3
"""Visualize Lost in the Middle experiment results."""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> dict:
    """Load results from JSON file."""
    with open(path) as f:
        return json.load(f)


def plot_accuracy_by_position(results: dict, output_dir: Path):
    """Plot accuracy vs position for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = results["config"]["positions"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(results["models"])))
    markers = ["o", "s", "^", "D", "v"]

    for i, (model_name, model_results) in enumerate(results["models"].items()):
        accuracies = [
            model_results["positions"][str(p)]["accuracy"] * 100 for p in positions
        ]

        ax.plot(
            positions,
            accuracies,
            marker=markers[i % len(markers)],
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=model_name,
        )

    ax.set_xlabel("Gold Document Position", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Lost in the Middle: Accuracy by Document Position", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(positions)

    # Set y-axis limits
    ax.set_ylim(0, 105)

    # Add middle indicator
    mid_pos = positions[len(positions) // 2]
    ax.axvline(x=mid_pos, color="red", linestyle="--", alpha=0.5, label="Middle")

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_position.png", dpi=150)
    plt.savefig(output_dir / "accuracy_by_position.pdf")
    print(f"Saved: accuracy_by_position.png/pdf")
    plt.close()


def plot_heatmap(results: dict, output_dir: Path):
    """Plot heatmap of accuracy by model and position."""
    fig, ax = plt.subplots(figsize=(10, 4))

    models = list(results["models"].keys())
    positions = results["config"]["positions"]

    # Build accuracy matrix
    matrix = []
    for model_name in models:
        row = [
            results["models"][model_name]["positions"][str(p)]["accuracy"] * 100
            for p in positions
        ]
        matrix.append(row)

    matrix = np.array(matrix)

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    # Labels
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"Pos {p}" for p in positions])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(positions)):
            text = f"{matrix[i, j]:.0f}%"
            color = "white" if matrix[i, j] < 50 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

    ax.set_title("Accuracy Heatmap by Model and Position", fontsize=14)

    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap.png", dpi=150)
    print(f"Saved: heatmap.png")
    plt.close()


def plot_delta_from_first(results: dict, output_dir: Path):
    """Plot accuracy delta from first position."""
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = results["config"]["positions"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(results["models"])))

    for i, (model_name, model_results) in enumerate(results["models"].items()):
        accuracies = [
            model_results["positions"][str(p)]["accuracy"] * 100 for p in positions
        ]

        # Calculate delta from first position
        baseline = accuracies[0]
        deltas = [acc - baseline for acc in accuracies]

        ax.bar(
            [p + i * 0.8 / len(results["models"]) - 0.4 for p in range(len(positions))],
            deltas,
            width=0.8 / len(results["models"]),
            color=colors[i],
            label=model_name,
            alpha=0.8,
        )

    ax.set_xlabel("Position Index", fontsize=12)
    ax.set_ylabel("Accuracy Change from Position 1 (%)", fontsize=12)
    ax.set_title("Accuracy Drop Relative to First Position", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"Pos {p}" for p in positions])
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "delta_from_first.png", dpi=150)
    print(f"Saved: delta_from_first.png")
    plt.close()


def print_summary_table(results: dict):
    """Print summary statistics."""
    positions = results["config"]["positions"]

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Header
    header = (
        "Model".ljust(15)
        + " | "
        + " | ".join([f"Pos {p:>2}" for p in positions])
        + " | Drop"
    )
    print(header)
    print("-" * len(header))

    for model_name, model_results in results["models"].items():
        accuracies = [
            model_results["positions"][str(p)]["accuracy"] * 100 for p in positions
        ]

        # Calculate drop (first - min)
        drop = max(accuracies) - min(accuracies)

        row = model_name.ljust(15) + " | "
        row += " | ".join([f"{acc:>5.1f}%" for acc in accuracies])
        row += f" | {drop:>4.1f}%"
        print(row)

    print("=" * 70)

    # Find middle position
    mid_idx = len(positions) // 2
    mid_pos = positions[mid_idx]

    print(f"\nMiddle position: {mid_pos}")
    print(f"Total documents: {results['config']['total_docs']}")
    print(f"Trials per position: {results['config']['trials_per_position']}")


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="results/results_v2.json",
        help="Input results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    # Load results
    input_path = Path(__file__).parent.parent / args.input
    results = load_results(input_path)

    output_dir = Path(__file__).parent.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("Generating visualizations...")
    plot_accuracy_by_position(results, output_dir)
    plot_heatmap(results, output_dir)
    plot_delta_from_first(results, output_dir)

    # Print summary
    print_summary_table(results)


if __name__ == "__main__":
    main()
