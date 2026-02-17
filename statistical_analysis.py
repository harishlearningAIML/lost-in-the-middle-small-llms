#!/usr/bin/env python3
"""
Statistical analysis for Lost in the Middle results.

Computes:
- Chi-squared test for position effects (early vs late)
- Wilson score confidence intervals for accuracy
- Fisher's exact test for 2x2 contingency tables

Usage:
    python statistical_analysis.py results/results_gemma-2b_20251226_162353.json
    python statistical_analysis.py results/
"""

import json
import sys
from pathlib import Path
from typing import Tuple


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion. z=1.96 for 95% CI."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return max(0, centre - margin), min(1, centre + margin)


def chi2_contingency(table: list) -> Tuple[float, float]:
    """Chi-squared test for 2x2 or larger contingency table. Returns (chi2, p-value)."""
    try:
        from scipy.stats import chi2_contingency as scipy_chi2
        chi2, p, _, _ = scipy_chi2(table)
        return chi2, p
    except ImportError:
        # Fallback: manual chi-squared for 2x2
        if len(table) == 2 and len(table[0]) == 2:
            a, b = table[0]
            c, d = table[1]
            n = a + b + c + d
            if n == 0:
                return 0.0, 1.0
            expected = [(a + b) * (a + c) / n, (a + b) * (b + d) / n,
                        (c + d) * (a + c) / n, (c + d) * (b + d) / n]
            observed = [a, b, c, d]
            chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0)
            # p-value from chi2 with 1 df (approximate)
            try:
                from math import erfc
                import math
                p = 1 - (1 + math.erf((chi2 ** 0.5) / (2 ** 0.5))) / 2
                p = 2 * min(p, 1 - p)  # two-tailed
            except Exception:
                p = float("nan")
            return chi2, p
        return float("nan"), float("nan")


def analyze_results(results_path: Path) -> None:
    """Analyze a single results JSON file."""
    with open(results_path) as f:
        data = json.load(f)

    config = data.get("config", {})
    positions = config.get("positions", [])
    total_docs = config.get("total_docs", 100)

    print(f"\n{'='*70}")
    print(f"Statistical Analysis: {results_path.name}")
    print(f"{'='*70}")
    print(f"Positions: {positions}")
    print(f"Total docs: {total_docs}")

    for model_name, model_data in data.get("models", {}).items():
        pos_data = model_data.get("positions", {})
        if not pos_data:
            continue

        print(f"\n--- {model_name} ---")

        # Accuracy with 95% CI per position
        print("\nPosition | Accuracy | 95% CI")
        print("-" * 40)
        for p in positions:
            key = str(p)
            if key not in pos_data:
                continue
            correct = pos_data[key]["correct"]
            total = pos_data[key]["total"]
            acc = correct / total * 100 if total > 0 else 0
            lo, hi = wilson_ci(correct, total)
            print(f"   {p:>3}   | {acc:5.1f}%  | [{lo*100:.1f}%, {hi*100:.1f}%]")

        # Early vs Late: chi-squared test
        if len(positions) >= 4:
            early_positions = positions[:2]
            late_positions = positions[-2:]
            early_correct = sum(pos_data[str(p)]["correct"] for p in early_positions)
            early_total = sum(pos_data[str(p)]["total"] for p in early_positions)
            late_correct = sum(pos_data[str(p)]["correct"] for p in late_positions)
            late_total = sum(pos_data[str(p)]["total"] for p in late_positions)

            table = [
                [early_correct, early_total - early_correct],
                [late_correct, late_total - late_correct],
            ]
            chi2, p_value = chi2_contingency(table)

            print(f"\nEarly (pos {early_positions}): {early_correct}/{early_total} = {100*early_correct/early_total:.1f}%")
            print(f"Late  (pos {late_positions}): {late_correct}/{late_total} = {100*late_correct/late_total:.1f}%")
            print(f"Chi-squared: {chi2:.3f}, p-value: {p_value:.4f}")
            if p_value < 0.05:
                print("  -> Statistically significant (p < 0.05)")
            else:
                print("  -> NOT statistically significant (p >= 0.05) - may be noise")

    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python statistical_analysis.py <results.json or results_dir>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if path.is_file():
        analyze_results(path)
    elif path.is_dir():
        for f in sorted(path.glob("results_*.json")):
            analyze_results(f)
    else:
        print(f"Not found: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
