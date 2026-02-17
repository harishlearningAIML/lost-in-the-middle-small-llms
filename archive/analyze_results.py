#!/usr/bin/env python3
"""Statistical analysis of Lost in the Middle experiment results."""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict

def load_results(filepath):
    """Load results JSON file."""
    with open(filepath) as f:
        return json.load(f)

def extract_accuracies(results):
    """Extract accuracy data by position for a model."""
    model_key = list(results["models"].keys())[0]
    model_data = results["models"][model_key]
    positions = results["config"]["positions"]
    
    accuracies = {}
    for pos in positions:
        pos_str = str(pos)
        if pos_str in model_data["positions"]:
            accuracies[pos] = model_data["positions"][pos_str]["accuracy"]
    
    return accuracies, positions

def analyze_trend(positions, accuracies):
    """Analyze if there's a significant trend (increasing/decreasing)."""
    pos_array = np.array(positions)
    acc_array = np.array([accuracies[p] for p in positions])
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(pos_array, acc_array)
    
    return {
        "slope": slope,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "trend": "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
    }

def compare_two_position_groups_fisher(model_raw_results, group1_positions, group2_positions):
    """
    Compares two groups of positions using Fisher's exact test.
    Input: model_raw_results (list of dicts from "raw_results"), two lists of positions.
    """
    group1_correct = 0
    group1_total = 0
    group2_correct = 0
    group2_total = 0

    for result in model_raw_results:
        pos = result["position"]
        if pos in group1_positions:
            group1_total += 1
            if result["correct"]:
                group1_correct += 1
        elif pos in group2_positions:
            group2_total += 1
            if result["correct"]:
                group2_correct += 1

    # Check for empty groups
    if group1_total == 0 or group2_total == 0:
        return None

    group1_incorrect = group1_total - group1_correct
    group2_incorrect = group2_total - group2_correct

    # Contingency table for Fisher's exact test
    # [[group1_correct, group1_incorrect],
    #  [group2_correct, group2_incorrect]]
    table = [[group1_correct, group1_incorrect],
             [group2_correct, group2_incorrect]]

    oddsratio, p_value = stats.fisher_exact(table)

    group1_accuracy = group1_correct / group1_total
    group2_accuracy = group2_correct / group2_total
    difference = group2_accuracy - group1_accuracy

    return {
        "group1_positions": group1_positions,
        "group2_positions": group2_positions,
        "group1_accuracy": group1_accuracy,
        "group2_accuracy": group2_accuracy,
        "difference": difference,
        "odds_ratio": oddsratio,
        "p_value": p_value,
        "significant": p_value < 0.05
    }

def analyze_model(results, model_name):
    """Comprehensive analysis for one model."""
    accuracies, positions = extract_accuracies(results)
    
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")
    
    # Print accuracy by position
    print("\nAccuracy by Position:")
    print("-" * 50)
    for pos in positions:
        acc = accuracies[pos] * 100
        print(f"  Position {pos:3d}: {acc:5.1f}%")
    
    # Trend analysis
    trend = analyze_trend(positions, accuracies)
    print(f"\nTrend Analysis:")
    print(f"  Slope: {trend['slope']:.6f} (positive = better at end)")
    print(f"  R²: {trend['r_squared']:.3f}")
    print(f"  P-value: {trend['p_value']:.4f}")
    print(f"  Trend: {trend['trend']}")
    print(f"  Significant trend: {'Yes' if trend['p_value'] < 0.05 else 'No'}")
    
    # Early vs Late comparison (using Fisher's Exact Test)
    # Define early (first 2 positions) and late (last 3 positions)
    early_positions = positions[:2]
    late_positions = positions[-3:]

    model_data = results["models"][model_name] # Access model_data here
    comparison = compare_two_position_groups_fisher(model_data["raw_results"], early_positions, late_positions)
    if comparison:
        print(f"\nEarly vs Late Comparison (Fisher's Exact Test):")
        print(f"  Early positions {comparison['group1_positions']}: {comparison['group1_accuracy']*100:.1f}%")
        print(f"  Late positions {comparison['group2_positions']}: {comparison['group2_accuracy']*100:.1f}%")
        print(f"  Difference (Late - Early): {comparison['difference']*100:+.1f}%")
        print(f"  Odds Ratio: {comparison['odds_ratio']:.3f}")
        print(f"  P-value: {comparison['p_value']:.4f}")
        print(f"  Statistically significant: {'Yes' if comparison['significant'] else 'No'}")
    
    # Check for U-curve pattern
    print(f"\nU-Curve Pattern Check:")
    first_acc = accuracies[positions[0]]
    mid_acc = accuracies[positions[len(positions)//2]]
    last_acc = accuracies[positions[-1]]
    
    u_curve_score = (first_acc + last_acc) / 2 - mid_acc
    print(f"  First position accuracy: {first_acc*100:.1f}%")
    print(f"  Middle position accuracy: {mid_acc*100:.1f}%")
    print(f"  Last position accuracy: {last_acc*100:.1f}%")
    print(f"  U-curve score (higher = more U-shaped): {u_curve_score*100:.1f}%")
    print(f"  Pattern: {'U-curve' if u_curve_score > 0.05 else 'Recency bias' if trend['slope'] > 0 else 'Primacy bias'}")
    
    return {
        "accuracies": accuracies,
        "trend": trend,
        "comparison": comparison,
        "u_curve_score": u_curve_score
    }

def main():
    results_dir = Path(__file__).parent / "results"
    
    files = {
        "Gemma-2B": results_dir / "results_gemma-2b_20251226_162353.json",
        "Gemma-4B": results_dir / "results_gemma-4b_20251226_165033.json",
        "Llama-3B": results_dir / "results_llama-3b_20251226_173208.json",
    }
    
    all_analyses = {}
    
    for model_name, filepath in files.items():
        if filepath.exists():
            results = load_results(filepath)
            analysis = analyze_model(results, model_name)
            all_analyses[model_name] = analysis
        else:
            print(f"\nWarning: {filepath} not found")
    
    # Cross-model comparison
    print(f"\n{'='*70}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*70}")
    
    print("\nTrend Consistency:")
    for model_name, analysis in all_analyses.items():
        trend = analysis["trend"]
        print(f"  {model_name:12s}: {trend['trend']:10s} (slope={trend['slope']:.6f}, p={trend['p_value']:.4f})")
    
    print("\nEarly vs Late Improvement:")
    for model_name, analysis in all_analyses.items():
        if analysis["comparison"]:
            comp = analysis["comparison"]
            sig = "*" if comp["significant"] else " "
            print(f"  {model_name:12s}: {comp['difference']*100:+6.1f}%{sig} (p={comp['p_value']:.4f})")
    
    print("\nU-Curve vs Recency Pattern:")
    for model_name, analysis in all_analyses.items():
        score = analysis["u_curve_score"]
        pattern = "U-curve" if score > 0.05 else "Recency bias"
        print(f"  {model_name:12s}: {pattern:15s} (score={score*100:.1f}%)")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Overall conclusion
    all_slopes_positive = all(a["trend"]["slope"] > 0 for a in all_analyses.values())
    all_significant = all(a["trend"]["p_value"] < 0.05 for a in all_analyses.values())
    
    print(f"\nAll models show recency bias: {all_slopes_positive}")
    print(f"All trends statistically significant: {all_significant}")
    
    if all_slopes_positive:
        print("\n✓ CONCLUSION: Consistent recency bias across all models")
        print("  Small models perform better when information is at the END")
        print("  This contradicts the 'Lost in the Middle' U-curve pattern")
    else:
        print("\n⚠ Mixed results - some models show different patterns")

if __name__ == "__main__":
    main()
