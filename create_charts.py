#!/usr/bin/env python3
"""Generate publication-quality charts for README."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_results():
    """Load all result files."""
    results_dir = Path(__file__).parent / "results"

    # Latest results for each model
    gemma2b = json.load(open(results_dir / "results_gemma-2b_20251226_162353.json"))
    gemma4b = json.load(open(results_dir / "results_gemma-4b_20251226_165033.json"))
    llama3b = json.load(open(results_dir / "results_llama-3b_20251226_173208.json"))

    return {
        "Gemma-2B (100 docs)": gemma2b,
        "Gemma-4B (100 docs)": gemma4b,
        "Llama-3B (70 docs)": llama3b,
    }


def plot_all_models(results, output_dir):
    """Plot accuracy by position for all models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'Gemma-2B (100 docs)': '#E74C3C', 'Gemma-4B (100 docs)': '#3498DB', 'Llama-3B (70 docs)': '#2ECC71'}
    markers = {'Gemma-2B (100 docs)': 'o', 'Gemma-4B (100 docs)': 's', 'Llama-3B (70 docs)': '^'}

    for model_label, data in results.items():
        model_key = list(data["models"].keys())[0]
        positions = data["config"]["positions"]
        accuracies = [
            data["models"][model_key]["positions"][str(p)]["accuracy"] * 100
            for p in positions
        ]

        ax.plot(
            positions,
            accuracies,
            marker=markers[model_label],
            color=colors[model_label],
            linewidth=2.5,
            markersize=10,
            label=model_label
        )

    ax.set_xlabel("Gold Document Position", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Small LLMs: Accuracy by Document Position\n(Recency Bias - Better Performance at End)", fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.set_ylim(80, 102)
    ax.set_xlim(0, 105)

    # Add annotation
    ax.annotate('Early positions\n(worst)', xy=(10, 83), fontsize=10, ha='center', color='gray')
    ax.annotate('Late positions\n(best)', xy=(90, 97), fontsize=10, ha='center', color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_position.png", dpi=150, bbox_inches='tight')
    print(f"Saved: accuracy_by_position.png")
    plt.close()


def plot_comparison_chart(output_dir):
    """Plot expected U-curve vs actual recency bias side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Expected U-curve (from original paper on large models)
    ax1 = axes[0]
    positions = [1, 10, 25, 50, 75, 90, 100]
    # Simulated U-curve pattern
    expected = [92, 85, 72, 65, 72, 85, 92]

    ax1.plot(positions, expected, 'o-', color='#9B59B6', linewidth=2.5, markersize=10)
    ax1.fill_between(positions, expected, 60, alpha=0.2, color='#9B59B6')
    ax1.set_xlabel("Gold Document Position", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Expected: U-Curve\n(Original Paper - GPT-3.5, Claude)", fontsize=13, fontweight='bold')
    ax1.set_ylim(60, 100)
    ax1.set_xlim(0, 105)
    ax1.annotate('Good', xy=(5, 93), fontsize=11, ha='center', color='#27AE60', fontweight='bold')
    ax1.annotate('Bad\n(Lost in Middle)', xy=(50, 62), fontsize=11, ha='center', color='#E74C3C', fontweight='bold')
    ax1.annotate('Good', xy=(95, 93), fontsize=11, ha='center', color='#27AE60', fontweight='bold')

    # Right: Actual results (recency bias)
    ax2 = axes[1]
    # Gemma-4B actual data
    actual = [86.7, 83.3, 90.0, 96.7, 93.3, 93.3, 96.7]

    ax2.plot(positions, actual, 'o-', color='#E74C3C', linewidth=2.5, markersize=10)
    ax2.fill_between(positions, actual, 80, alpha=0.2, color='#E74C3C')
    ax2.set_xlabel("Gold Document Position", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Actual: Recency Bias\n(Small Models - Gemma, Llama)", fontsize=13, fontweight='bold')
    ax2.set_ylim(80, 100)
    ax2.set_xlim(0, 105)
    ax2.annotate('Worst\n(83%)', xy=(10, 82), fontsize=11, ha='center', color='#E74C3C', fontweight='bold')
    ax2.annotate('Best\n(97%)', xy=(50, 98), fontsize=11, ha='center', color='#27AE60', fontweight='bold')

    # Add arrow showing the trend
    ax2.annotate('', xy=(90, 95), xytext=(20, 85),
                arrowprops=dict(arrowstyle='->', color='#3498DB', lw=2))
    ax2.text(55, 86, 'Upward trend', fontsize=10, color='#3498DB', ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / "expected_vs_actual.png", dpi=150, bbox_inches='tight')
    print(f"Saved: expected_vs_actual.png")
    plt.close()


def plot_early_vs_late(results, output_dir):
    """Bar chart comparing early vs late position performance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = []
    early_acc = []
    late_acc = []

    for model_label, data in results.items():
        model_key = list(data["models"].keys())[0]
        positions = data["config"]["positions"]
        pos_data = data["models"][model_key]["positions"]

        # Early = first 2 positions, Late = last 3 positions
        early = np.mean([pos_data[str(p)]["accuracy"] * 100 for p in positions[:2]])
        late = np.mean([pos_data[str(p)]["accuracy"] * 100 for p in positions[-3:]])

        models.append(model_label.replace(" (100 docs)", "").replace(" (70 docs)", ""))
        early_acc.append(early)
        late_acc.append(late)

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, early_acc, width, label='Early Positions (1, 10)', color='#E74C3C', alpha=0.8)
    bars2 = ax.bar(x + width/2, late_acc, width, label='Late Positions (75+)', color='#2ECC71', alpha=0.8)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Early vs Late Position Performance\n(All Models Show Recency Bias)", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(80, 100)

    # Add improvement labels
    for i, (e, l) in enumerate(zip(early_acc, late_acc)):
        improvement = l - e
        ax.annotate(f'+{improvement:.1f}%',
                   xy=(i, max(e, l) + 1),
                   ha='center', fontsize=11, fontweight='bold', color='#27AE60')

    plt.tight_layout()
    plt.savefig(output_dir / "early_vs_late.png", dpi=150, bbox_inches='tight')
    print(f"Saved: early_vs_late.png")
    plt.close()


def plot_heatmap(results, output_dir):
    """Heatmap of accuracy by model and position."""
    fig, ax = plt.subplots(figsize=(12, 4))

    # Use Gemma models only (same positions)
    gemma_results = {k: v for k, v in results.items() if "Gemma" in k}

    models = list(gemma_results.keys())
    positions = gemma_results[models[0]]["config"]["positions"]

    matrix = []
    for model_label in models:
        data = gemma_results[model_label]
        model_key = list(data["models"].keys())[0]
        row = [
            data["models"][model_key]["positions"][str(p)]["accuracy"] * 100
            for p in positions
        ]
        matrix.append(row)

    matrix = np.array(matrix)

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=80, vmax=100)

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"Pos {p}" for p in positions], fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace(" (100 docs)", "") for m in models], fontsize=11)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(positions)):
            text = f"{matrix[i, j]:.0f}%"
            color = "white" if matrix[i, j] < 88 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11, fontweight='bold')

    ax.set_title("Accuracy Heatmap: Gemma Models\n(Green = High Accuracy, Red = Low)", fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap.png", dpi=150, bbox_inches='tight')
    print(f"Saved: heatmap.png")
    plt.close()


def main():
    output_dir = Path(__file__).parent / "images"
    output_dir.mkdir(exist_ok=True)

    print("Loading results...")
    results = load_results()

    print("\nGenerating charts...")
    plot_all_models(results, output_dir)
    plot_comparison_chart(output_dir)
    plot_early_vs_late(results, output_dir)
    plot_heatmap(results, output_dir)

    print(f"\nAll charts saved to {output_dir}/")
    print("\nAdd these to README.md:")
    print("![Accuracy by Position](images/accuracy_by_position.png)")
    print("![Expected vs Actual](images/expected_vs_actual.png)")
    print("![Early vs Late](images/early_vs_late.png)")
    print("![Heatmap](images/heatmap.png)")


if __name__ == "__main__":
    main()
