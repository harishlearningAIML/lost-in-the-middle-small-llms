"""
Visualize Results - Generate charts for Lost in the Middle experiment

Usage:
    python visualize.py                          # Use default results file
    python visualize.py --input results.json     # Specify input file
"""

import json
import argparse
from pathlib import Path

# Try matplotlib, fall back to text-based output
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed. Will generate text-based output.")


def load_results(path: str) -> dict:
    """Load results from JSON file"""
    with open(path, "r") as f:
        return json.load(f)


def plot_accuracy_by_position(results: dict, output_path: str = "results/position_accuracy.png"):
    """Generate the main U-shaped curve chart"""
    
    if not HAS_MATPLOTLIB:
        print_text_chart(results)
        return
    
    positions = results["config"]["positions"]
    
    # Colors for different models
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    
    plt.figure(figsize=(10, 6))
    
    for i, (model_name, model_data) in enumerate(results["models"].items()):
        if "error" in model_data:
            continue
            
        accuracies = [model_data["positions"][str(p)]["accuracy"] * 100 
                      for p in positions]
        
        plt.plot(
            positions, 
            accuracies, 
            marker='o', 
            linewidth=2,
            markersize=8,
            color=colors[i % len(colors)],
            label=model_name
        )
    
    plt.xlabel("Gold Document Position", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Lost in the Middle: Accuracy by Document Position", fontsize=14, fontweight='bold')
    
    plt.xticks(positions)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower center')
    
    # Add annotation for middle
    middle_pos = positions[len(positions) // 2]
    plt.axvline(x=middle_pos, color='gray', linestyle='--', alpha=0.5)
    plt.text(middle_pos + 0.3, 95, "Middle", fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved chart to {output_path}")


def plot_heatmap(results: dict, output_path: str = "results/heatmap.png"):
    """Generate heatmap of accuracy by model and position"""
    
    if not HAS_MATPLOTLIB:
        return
    
    import numpy as np
    
    positions = results["config"]["positions"]
    models = [m for m in results["models"].keys() if "error" not in results["models"][m]]
    
    if not models:
        print("No valid model results to plot")
        return
    
    # Build data matrix
    data = []
    for model_name in models:
        model_data = results["models"][model_name]
        row = [model_data["positions"][str(p)]["accuracy"] * 100 for p in positions]
        data.append(row)
    
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Labels
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"Pos {p}" for p in positions])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    
    # Add values in cells
    for i in range(len(models)):
        for j in range(len(positions)):
            val = data[i, j]
            color = 'white' if val < 50 else 'black'
            ax.text(j, i, f"{val:.0f}%", ha='center', va='center', color=color, fontsize=11)
    
    plt.colorbar(im, label='Accuracy (%)')
    plt.title("Accuracy Heatmap: Model × Position", fontsize=14, fontweight='bold')
    plt.xlabel("Document Position")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap to {output_path}")


def plot_delta_from_best(results: dict, output_path: str = "results/delta.png"):
    """Show accuracy drop from best position"""
    
    if not HAS_MATPLOTLIB:
        return
    
    positions = results["config"]["positions"]
    colors = ["#2563eb", "#dc2626", "#16a34a"]
    
    plt.figure(figsize=(10, 6))
    
    for i, (model_name, model_data) in enumerate(results["models"].items()):
        if "error" in model_data:
            continue
            
        accuracies = [model_data["positions"][str(p)]["accuracy"] * 100 
                      for p in positions]
        
        best = max(accuracies)
        deltas = [acc - best for acc in accuracies]
        
        plt.bar(
            [p + i*0.25 for p in positions],
            deltas,
            width=0.25,
            color=colors[i % len(colors)],
            label=model_name,
            alpha=0.8
        )
    
    plt.xlabel("Gold Document Position", fontsize=12)
    plt.ylabel("Accuracy Drop from Best (%)", fontsize=12)
    plt.title("Performance Drop by Position (vs. Best Position)", fontsize=14, fontweight='bold')
    
    plt.xticks(positions)
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved delta chart to {output_path}")


def print_text_chart(results: dict):
    """Fallback text-based visualization"""
    
    positions = results["config"]["positions"]
    
    print("\n" + "="*70)
    print("ACCURACY BY POSITION (Text Chart)")
    print("="*70)
    
    for model_name, model_data in results["models"].items():
        if "error" in model_data:
            print(f"\n{model_name}: ERROR")
            continue
            
        print(f"\n{model_name}:")
        
        for pos in positions:
            acc = model_data["positions"][str(pos)]["accuracy"]
            bar_len = int(acc * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"  Pos {pos:>2}: [{bar}] {acc:.1%}")


def generate_summary_stats(results: dict) -> dict:
    """Calculate summary statistics"""
    
    positions = results["config"]["positions"]
    stats = {}
    
    for model_name, model_data in results["models"].items():
        if "error" in model_data:
            continue
            
        accuracies = [model_data["positions"][str(p)]["accuracy"] for p in positions]
        
        best_idx = accuracies.index(max(accuracies))
        worst_idx = accuracies.index(min(accuracies))
        
        stats[model_name] = {
            "best_position": positions[best_idx],
            "best_accuracy": max(accuracies),
            "worst_position": positions[worst_idx],
            "worst_accuracy": min(accuracies),
            "drop": max(accuracies) - min(accuracies),
            "mean_accuracy": sum(accuracies) / len(accuracies)
        }
    
    return stats


def print_insights(results: dict):
    """Print key insights from the experiment"""
    
    stats = generate_summary_stats(results)
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    for model_name, s in stats.items():
        print(f"\n{model_name}:")
        print(f"  Best:  Position {s['best_position']} ({s['best_accuracy']:.1%})")
        print(f"  Worst: Position {s['worst_position']} ({s['worst_accuracy']:.1%})")
        print(f"  Drop:  {s['drop']:.1%} accuracy difference")
        print(f"  Mean:  {s['mean_accuracy']:.1%}")
    
    # Overall insights
    if stats:
        avg_drop = sum(s['drop'] for s in stats.values()) / len(stats)
        
        print("\n" + "-"*40)
        print(f"Average accuracy drop across models: {avg_drop:.1%}")
        
        # Check if middle is consistently worst
        middle_pos = results["config"]["positions"][len(results["config"]["positions"]) // 2]
        middle_worst_count = sum(1 for s in stats.values() 
                                  if s['worst_position'] == middle_pos)
        
        if middle_worst_count == len(stats):
            print(f"✓ All models performed worst at middle position ({middle_pos})")
        elif middle_worst_count > 0:
            print(f"⚠ {middle_worst_count}/{len(stats)} models performed worst at middle")


def main():
    parser = argparse.ArgumentParser(description="Visualize Lost in the Middle results")
    parser.add_argument("--input", type=str, default="results/results.json", help="Input results file")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    # Load results
    try:
        results = load_results(args.input)
    except FileNotFoundError:
        print(f"Results file not found: {args.input}")
        print("Run the experiment first: python run_experiment.py")
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Generate visualizations
    plot_accuracy_by_position(results, f"{args.output_dir}/position_accuracy.png")
    plot_heatmap(results, f"{args.output_dir}/heatmap.png")
    plot_delta_from_best(results, f"{args.output_dir}/delta.png")
    
    # Print insights
    print_insights(results)
    
    # Save stats
    stats = generate_summary_stats(results)
    with open(f"{args.output_dir}/summary_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved summary stats to {args.output_dir}/summary_stats.json")


if __name__ == "__main__":
    main()
