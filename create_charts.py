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

    # Latest results for each model (Feb 2026, 72 trials)
    gemma2b = json.load(open(results_dir / "results_gemma-2b_20260211_091248.json"))
    gemma4b = json.load(open(results_dir / "results_gemma-4b_20260211_094409.json"))
    llama3b = json.load(open(results_dir / "results_llama-3b_20260211_105815.json"))

    return {
        "Gemma-2B (100 docs)": gemma2b,
        "Gemma-4B (100 docs)": gemma4b,
        "Llama-3B (70 docs)": llama3b,
    }


def plot_all_models(results, output_dir):
    """Plot accuracy by position for all models.
    Uses % of context (0-100) for fair comparison: Llama 70 docs vs Gemma 100 docs.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'Gemma-2B (100 docs)': '#E74C3C', 'Gemma-4B (100 docs)': '#3498DB', 'Llama-3B (70 docs)': '#2ECC71'}
    markers = {'Gemma-2B (100 docs)': 'o', 'Gemma-4B (100 docs)': 's', 'Llama-3B (70 docs)': '^'}

    for model_label, data in results.items():
        model_key = list(data["models"].keys())[0]
        positions = data["config"]["positions"]
        total_docs = data["config"].get("total_docs", 100)
        # Normalize to % of context for fair cross-model comparison
        pct_positions = [100 * p / total_docs for p in positions]
        accuracies = [
            data["models"][model_key]["positions"][str(p)]["accuracy"] * 100
            for p in positions
        ]

        ax.plot(
            pct_positions,
            accuracies,
            marker=markers[model_label],
            color=colors[model_label],
            linewidth=2.5,
            markersize=10,
            label=model_label
        )

    ax.set_xlabel("Gold Document Position (% of Context)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Small LLMs: Accuracy by Document Position\n(Normalized - Fair comparison across context lengths)", fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.set_ylim(80, 102)
    ax.set_xlim(0, 105)

    ax.annotate('Early\n(worst)', xy=(10, 83), fontsize=10, ha='center', color='gray')
    ax.annotate('Late\n(best)', xy=(90, 97), fontsize=10, ha='center', color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_position.png", dpi=150, bbox_inches='tight')
    print(f"Saved: accuracy_by_position.png")
    plt.close()


def plot_comparison_chart(results, output_dir):
    """Plot expected U-curve vs actual results for each model."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Left: Expected U-curve (from original paper on large models)
    ax1 = axes[0]
    positions_expected = [1, 10, 25, 50, 75, 90, 100]
    # Simulated U-curve pattern
    expected = [92, 85, 72, 65, 72, 85, 92]

    ax1.plot(positions_expected, expected, 'o-', color='#9B59B6', linewidth=2.5, markersize=10)
    ax1.fill_between(positions_expected, expected, 60, alpha=0.2, color='#9B59B6')
    ax1.set_xlabel("Gold Document Position", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Expected: U-Curve\n(Original Paper - GPT-3.5, Claude)", fontsize=13, fontweight='bold')
    ax1.set_ylim(60, 100)
    ax1.set_xlim(0, 105)
    ax1.annotate('Good', xy=(5, 93), fontsize=11, ha='center', color='#27AE60', fontweight='bold')
    ax1.annotate('Bad\n(Lost in Middle)', xy=(50, 62), fontsize=11, ha='center', color='#E74C3C', fontweight='bold')
    ax1.annotate('Good', xy=(95, 93), fontsize=11, ha='center', color='#27AE60', fontweight='bold')

    # Model colors and info
    model_configs = [
        ("Gemma-2B (100 docs)", '#E74C3C', "Gemma-2B: Recency Bias\n(Worst at beginning)"),
        ("Gemma-4B (100 docs)", '#3498DB', "Gemma-4B: Weak Lost in Middle\n(Worst at position 50)"),
        ("Llama-3B (70 docs)", '#2ECC71', "Llama-3B: Flat/Stable\n(No significant pattern)"),
    ]

    for idx, (model_label, color, title) in enumerate(model_configs):
        ax = axes[idx + 1]
        data = results[model_label]
        model_key = list(data["models"].keys())[0]
        positions = data["config"]["positions"]
        total_docs = data["config"].get("total_docs", 100)

        accuracies = [
            data["models"][model_key]["positions"][str(p)]["accuracy"] * 100
            for p in positions
        ]

        ax.plot(positions, accuracies, 'o-', color=color, linewidth=2.5, markersize=10)
        ax.fill_between(positions, accuracies, min(accuracies) - 5, alpha=0.2, color=color)
        ax.set_xlabel("Gold Document Position", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(80, 100)
        ax.set_xlim(0, max(positions) + 5)

        # Annotate min and max
        min_idx = np.argmin(accuracies)
        max_idx = np.argmax(accuracies)
        ax.annotate(f'Worst\n({accuracies[min_idx]:.0f}%)',
                   xy=(positions[min_idx], accuracies[min_idx] - 1),
                   fontsize=9, ha='center', color='#E74C3C', fontweight='bold')
        ax.annotate(f'Best\n({accuracies[max_idx]:.0f}%)',
                   xy=(positions[max_idx], accuracies[max_idx] + 1),
                   fontsize=9, ha='center', color='#27AE60', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "expected_vs_actual.png", dpi=150, bbox_inches='tight')
    print(f"Saved: expected_vs_actual.png")
    plt.close()


def plot_early_vs_late(results, output_dir):
    """Bar chart comparing early vs late position performance.
    Uses % of context: early = first 20%, late = last 30% (fair across 70 vs 100 docs).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = []
    early_acc = []
    late_acc = []

    for model_label, data in results.items():
        model_key = list(data["models"].keys())[0]
        positions = data["config"]["positions"]
        total_docs = data["config"].get("total_docs", 100)
        pos_data = data["models"][model_key]["positions"]

        # Early = positions in first 20% of context, Late = positions in last 30%
        pct_positions = [(p, 100 * p / total_docs) for p in positions]
        early_positions = [p for p, pct in pct_positions if pct <= 20]
        late_positions = [p for p, pct in pct_positions if pct >= 70]
        if not early_positions:
            early_positions = positions[:2]
        if not late_positions:
            late_positions = positions[-2:]

        early = np.mean([pos_data[str(p)]["accuracy"] * 100 for p in early_positions])
        late = np.mean([pos_data[str(p)]["accuracy"] * 100 for p in late_positions])

        models.append(model_label.replace(" (100 docs)", "").replace(" (70 docs)", ""))
        early_acc.append(early)
        late_acc.append(late)

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, early_acc, width, label='Early (≤20% of context)', color='#E74C3C', alpha=0.8)
    bars2 = ax.bar(x + width/2, late_acc, width, label='Late (≥70% of context)', color='#2ECC71', alpha=0.8)

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
    """Create two separate heatmaps - one for 100-doc models, one for 70-doc model."""
    # Heatmap 1: Gemma models (100 docs)
    fig, ax = plt.subplots(figsize=(12, 4))

    gemma_results = {k: v for k, v in results.items() if "Gemma" in k}
    models = list(gemma_results.keys())
    positions = [1, 10, 25, 50, 75, 90, 100]  # 100-doc positions

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

    for i in range(len(models)):
        for j in range(len(positions)):
            text = f"{matrix[i, j]:.0f}%"
            color = "white" if matrix[i, j] < 88 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11, fontweight='bold')

    ax.set_xlabel("Gold Document Position (out of 100 docs)", fontsize=12)
    ax.set_title("Accuracy Heatmap: Gemma Models (100 Documents)\n(Green = High Accuracy, Red = Low)", fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_100docs.png", dpi=150, bbox_inches='tight')
    print(f"Saved: heatmap_100docs.png")
    plt.close()

    # Heatmap 2: Llama model (70 docs)
    fig, ax = plt.subplots(figsize=(12, 3))

    llama_data = results["Llama-3B (70 docs)"]
    model_key = list(llama_data["models"].keys())[0]
    positions_70 = llama_data["config"]["positions"]  # [7, 14, 35, 49, 63, 70]

    matrix = []
    row = [
        llama_data["models"][model_key]["positions"][str(p)]["accuracy"] * 100
        for p in positions_70
    ]
    matrix.append(row)

    matrix = np.array(matrix)

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=80, vmax=100)

    ax.set_xticks(range(len(positions_70)))
    ax.set_xticklabels([f"Pos {p}" for p in positions_70], fontsize=11)
    ax.set_yticks([0])
    ax.set_yticklabels(["Llama-3B"], fontsize=11)

    for j in range(len(positions_70)):
        text = f"{matrix[0, j]:.0f}%"
        color = "white" if matrix[0, j] < 88 else "black"
        ax.text(j, 0, text, ha="center", va="center", color=color, fontsize=11, fontweight='bold')

    ax.set_xlabel("Gold Document Position (out of 70 docs)", fontsize=12)
    ax.set_title("Accuracy Heatmap: Llama-3B (70 Documents)\n(Green = High Accuracy, Red = Low)", fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_70docs.png", dpi=150, bbox_inches='tight')
    print(f"Saved: heatmap_70docs.png")
    plt.close()


def main():
    output_dir = Path(__file__).parent / "images"
    output_dir.mkdir(exist_ok=True)

    print("Loading results...")
    results = load_results()

    print("\nGenerating charts...")
    plot_all_models(results, output_dir)
    plot_comparison_chart(results, output_dir)
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
