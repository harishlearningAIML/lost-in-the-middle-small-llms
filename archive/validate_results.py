#!/usr/bin/env python3
"""
Comprehensive validation script for Lost in the Middle experiment results.
Verifies data integrity, accuracy calculations, and statistical claims.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np
from scipy import stats


class ResultValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []

    def log_error(self, msg: str):
        self.errors.append(f"❌ ERROR: {msg}")

    def log_warning(self, msg: str):
        self.warnings.append(f"⚠️  WARNING: {msg}")

    def log_info(self, msg: str):
        self.info.append(f"✓ {msg}")

    def validate_result_file(self, filepath: Path) -> Dict[str, Any]:
        """Load and validate a single result file."""
        print(f"\n{'='*80}")
        print(f"Validating: {filepath.name}")
        print(f"{'='*80}")

        if not filepath.exists():
            self.log_error(f"File not found: {filepath}")
            return None

        with open(filepath) as f:
            data = json.load(f)

        # Validate structure
        if "config" not in data:
            self.log_error("Missing 'config' section")
            return None

        if "models" not in data:
            self.log_error("Missing 'models' section")
            return None

        config = data["config"]

        # Validate config
        required_config = ["positions", "total_docs", "trials_per_position", "max_new_tokens", "temperature"]
        for key in required_config:
            if key not in config:
                self.log_error(f"Missing config key: {key}")

        self.log_info(f"Config: {len(config['positions'])} positions, {config['total_docs']} docs, {config['trials_per_position']} trials/position")

        # Validate each model
        for model_name, model_data in data["models"].items():
            self.validate_model_data(model_name, model_data, config)

        return data

    def validate_model_data(self, model_name: str, model_data: Dict, config: Dict):
        """Validate a single model's data."""
        print(f"\n--- Model: {model_name} ---")

        positions = config["positions"]
        trials_per_pos = config["trials_per_position"]

        # Check we have position summary data
        if "positions" not in model_data:
            self.log_error(f"{model_name}: Missing 'positions' summary")
            return

        # Check we have raw results
        if "raw_results" not in model_data:
            self.log_error(f"{model_name}: Missing 'raw_results'")
            return

        raw_results = model_data["raw_results"]
        position_summary = model_data["positions"]

        # Validate expected number of results
        expected_total = len(positions) * trials_per_pos
        actual_total = len(raw_results)

        if actual_total != expected_total:
            self.log_error(f"{model_name}: Expected {expected_total} results, got {actual_total}")
        else:
            self.log_info(f"{model_name}: Correct number of results ({actual_total})")

        # Validate each position has correct number of trials
        results_by_position = defaultdict(list)
        for result in raw_results:
            results_by_position[result["position"]].append(result)

        for pos in positions:
            count = len(results_by_position[pos])
            if count != trials_per_pos:
                self.log_error(f"{model_name} position {pos}: Expected {trials_per_pos} trials, got {count}")
            else:
                self.log_info(f"{model_name} position {pos}: {count} trials")

        # RECALCULATE ACCURACY from raw results
        print(f"\n  Accuracy Verification:")
        accuracy_matches = True

        for pos in positions:
            results_at_pos = results_by_position[pos]

            # Recalculate accuracy
            correct_count = sum(1 for r in results_at_pos if r["correct"])
            total_count = len(results_at_pos)
            calculated_accuracy = correct_count / total_count if total_count > 0 else 0

            # Get reported accuracy
            pos_str = str(pos)
            if pos_str not in position_summary:
                self.log_error(f"{model_name}: Missing summary for position {pos}")
                continue

            reported = position_summary[pos_str]
            reported_accuracy = reported["accuracy"]
            reported_correct = reported["correct"]
            reported_total = reported["total"]

            # Check if they match
            if abs(calculated_accuracy - reported_accuracy) > 0.001:
                self.log_error(f"{model_name} pos {pos}: Accuracy mismatch! Calculated={calculated_accuracy:.4f}, Reported={reported_accuracy:.4f}")
                accuracy_matches = False
            elif correct_count != reported_correct:
                self.log_error(f"{model_name} pos {pos}: Correct count mismatch! Calculated={correct_count}, Reported={reported_correct}")
                accuracy_matches = False
            elif total_count != reported_total:
                self.log_error(f"{model_name} pos {pos}: Total count mismatch! Calculated={total_count}, Reported={reported_total}")
                accuracy_matches = False
            else:
                self.log_info(f"  Pos {pos:3d}: {calculated_accuracy*100:5.1f}% ({correct_count}/{total_count}) ✓")

        if accuracy_matches:
            self.log_info(f"{model_name}: All accuracy calculations verified ✓")

        # Validate individual result entries
        for i, result in enumerate(raw_results):
            required_fields = ["correct", "response", "extracted", "gold_answer", "qa_id", "position"]
            for field in required_fields:
                if field not in result:
                    self.log_error(f"{model_name} result {i}: Missing field '{field}'")

        return results_by_position

    def calculate_statistics(self, results_data: Dict[str, Any], model_name: str) -> Dict:
        """Recalculate statistical metrics."""
        model_data = results_data["models"][model_name]
        positions = results_data["config"]["positions"]

        # Get accuracy values
        accuracies = []
        position_vals = []

        for pos in positions:
            pos_str = str(pos)
            if pos_str in model_data["positions"]:
                accuracies.append(model_data["positions"][pos_str]["accuracy"])
                position_vals.append(pos)

        # Linear regression to detect trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(position_vals, accuracies)

        # Early vs Late comparison
        early_positions = [p for p in position_vals if p <= 10]
        late_positions = [p for p in position_vals if p >= 75] if max(position_vals) >= 75 else [p for p in position_vals if p >= max(position_vals) * 0.75]

        early_acc = [model_data["positions"][str(p)]["accuracy"] for p in early_positions]
        late_acc = [model_data["positions"][str(p)]["accuracy"] for p in late_positions]

        if len(early_acc) >= 2 and len(late_acc) >= 2:
            t_stat, t_pval = stats.ttest_ind(early_acc, late_acc)
        else:
            t_stat, t_pval = None, None

        # U-curve score
        first_acc = accuracies[0]
        last_acc = accuracies[-1]
        middle_acc = np.mean(accuracies[len(accuracies)//3:2*len(accuracies)//3])
        u_curve_score = (first_acc + last_acc) / 2 - middle_acc

        return {
            "slope": slope,
            "p_value": p_value,
            "r_squared": r_value**2,
            "early_mean": np.mean(early_acc),
            "late_mean": np.mean(late_acc),
            "t_pvalue": t_pval,
            "u_curve_score": u_curve_score,
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "range": max(accuracies) - min(accuracies)
        }

    def print_summary(self):
        """Print validation summary."""
        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}\n")

        if self.errors:
            print(f"❌ ERRORS ({len(self.errors)}):")
            for err in self.errors:
                print(f"  {err}")
        else:
            print("✓ No errors found!")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warn in self.warnings:
                print(f"  {warn}")

        print(f"\n✓ INFO ({len(self.info)}):")
        for info in self.info[:10]:  # Show first 10
            print(f"  {info}")
        if len(self.info) > 10:
            print(f"  ... and {len(self.info) - 10} more")

        print(f"\n{'='*80}")
        if self.errors:
            print("❌ VALIDATION FAILED")
            return False
        else:
            print("✓ VALIDATION PASSED")
            return True


def main():
    validator = ResultValidator()

    results_dir = Path("results")

    # Find the latest results for each model
    model_files = {
        "gemma-2b": "results/results_gemma-2b_20251226_162353.json",
        "gemma-4b": "results/results_gemma-4b_20251226_165033.json",
        "llama-3b": "results/results_llama-3b_20251226_173208.json"
    }

    all_data = {}

    for model_name, filepath in model_files.items():
        path = Path(filepath)
        data = validator.validate_result_file(path)
        if data:
            all_data[model_name] = data

    # Calculate and verify statistics
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS VERIFICATION")
    print(f"{'='*80}\n")

    for model_name, data in all_data.items():
        print(f"\n--- {model_name.upper()} ---")
        model_key = model_name.replace("-", "_")

        stats_results = validator.calculate_statistics(data, model_name)

        print(f"  Accuracy Range: {stats_results['min_accuracy']*100:.1f}% - {stats_results['max_accuracy']*100:.1f}% (Δ={stats_results['range']*100:.1f}%)")
        print(f"  Linear Trend: slope={stats_results['slope']:.6f}, p={stats_results['p_value']:.4f}, R²={stats_results['r_squared']:.4f}")

        if stats_results['p_value'] < 0.05:
            direction = "increasing" if stats_results['slope'] > 0 else "decreasing"
            print(f"  → Statistically significant {direction} trend (p < 0.05) ✓")
        else:
            print(f"  → No statistically significant trend (p >= 0.05)")

        print(f"  Early positions (mean): {stats_results['early_mean']*100:.1f}%")
        print(f"  Late positions (mean): {stats_results['late_mean']*100:.1f}%")

        if stats_results['t_pvalue'] is not None:
            print(f"  Early vs Late t-test: p={stats_results['t_pvalue']:.4f}")

        print(f"  U-curve score: {stats_results['u_curve_score']:.4f}")
        if stats_results['u_curve_score'] > 0:
            print(f"  → Exhibits U-curve pattern (good at ends, bad in middle)")
        else:
            print(f"  → Exhibits recency bias (better at end than start)")

    # Verify claims from README
    print(f"\n{'='*80}")
    print("CLAIMS VERIFICATION")
    print(f"{'='*80}\n")

    # Claim 1: Gemma-2B shows 86.7% -> 93.3%
    gemma2b_data = all_data.get("gemma-2b")
    if gemma2b_data:
        pos1 = gemma2b_data["models"]["gemma-2b"]["positions"]["1"]["accuracy"]
        pos100 = gemma2b_data["models"]["gemma-2b"]["positions"]["100"]["accuracy"]

        if abs(pos1 - 0.8667) < 0.01 and abs(pos100 - 0.9333) < 0.01:
            validator.log_info(f"Gemma-2B: Position 1={pos1*100:.1f}%, Position 100={pos100*100:.1f}% ✓")
        else:
            validator.log_error(f"Gemma-2B accuracy mismatch: Got {pos1*100:.1f}% -> {pos100*100:.1f}%")

    # Claim 2: Gemma-4B shows 83.3% -> 96.7%
    gemma4b_data = all_data.get("gemma-4b")
    if gemma4b_data:
        pos1 = gemma4b_data["models"]["gemma-4b"]["positions"]["1"]["accuracy"]
        pos100 = gemma4b_data["models"]["gemma-4b"]["positions"]["100"]["accuracy"]

        # Note: The position keys might be different
        positions_available = list(gemma4b_data["models"]["gemma-4b"]["positions"].keys())
        first_pos = min([int(p) for p in positions_available])
        last_pos = max([int(p) for p in positions_available])

        first_acc = gemma4b_data["models"]["gemma-4b"]["positions"][str(first_pos)]["accuracy"]
        last_acc = gemma4b_data["models"]["gemma-4b"]["positions"][str(last_pos)]["accuracy"]

        improvement = (last_acc - first_acc) * 100
        validator.log_info(f"Gemma-4B: Position {first_pos}={first_acc*100:.1f}%, Position {last_pos}={last_acc*100:.1f}% (Δ=+{improvement:.1f}%)")

        if improvement > 10:
            validator.log_info(f"Gemma-4B shows strong recency bias (+{improvement:.1f}%) ✓")
        else:
            validator.log_warning(f"Gemma-4B improvement is {improvement:.1f}%, less than reported")

    # Claim 3: Llama-3B shows 93.3% -> 95.0%
    llama_data = all_data.get("llama-3b")
    if llama_data:
        positions_available = list(llama_data["models"]["llama-3b"]["positions"].keys())
        first_pos = min([int(p) for p in positions_available])
        last_pos = max([int(p) for p in positions_available])

        first_acc = llama_data["models"]["llama-3b"]["positions"][str(first_pos)]["accuracy"]
        last_acc = llama_data["models"]["llama-3b"]["positions"][str(last_pos)]["accuracy"]

        improvement = (last_acc - first_acc) * 100
        validator.log_info(f"Llama-3B: Position {first_pos}={first_acc*100:.1f}%, Position {last_pos}={last_acc*100:.1f}% (Δ={improvement:+.1f}%)")

    # Check for duplicate QA IDs in same position
    for model_name, data in all_data.items():
        model_data = data["models"][model_name]
        seen = set()
        for result in model_data["raw_results"]:
            key = (result["qa_id"], result["position"])
            if key in seen:
                validator.log_error(f"{model_name}: Duplicate result for {result['qa_id']} at position {result['position']}")
            seen.add(key)

    # Verify determinism: same question at different positions should have different answers
    # (unless model is perfectly consistent, but we expect position effects)
    for model_name, data in all_data.items():
        model_data = data["models"][model_name]
        results_by_qa = defaultdict(list)

        for result in model_data["raw_results"]:
            results_by_qa[result["qa_id"]].append(result)

        # Check if we have at least some variance across positions
        total_variance = 0
        for qa_id, results in results_by_qa.items():
            correctness_values = [r["correct"] for r in results]
            if len(set(correctness_values)) > 1:
                total_variance += 1

        variance_pct = (total_variance / len(results_by_qa)) * 100
        validator.log_info(f"{model_name}: {variance_pct:.1f}% of questions show position-dependent results")

        if variance_pct < 10:
            validator.log_warning(f"{model_name}: Very low variance across positions - possible issue with experiment setup")

    return validator


def main():
    validator = ResultValidator()

    results_dir = Path("results")

    # Find the latest results for each model
    model_files = {
        "gemma-2b": Path("results/results_gemma-2b_20251226_162353.json"),
        "gemma-4b": Path("results/results_gemma-4b_20251226_165033.json"),
        "llama-3b": Path("results/results_llama-3b_20251226_173208.json")
    }

    all_data = {}

    for model_name, filepath in model_files.items():
        data = validator.validate_result_file(filepath)
        if data:
            all_data[model_name] = data

    # Calculate and verify statistics
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS VERIFICATION")
    print(f"{'='*80}\n")

    for model_name, data in all_data.items():
        print(f"\n--- {model_name.upper()} ---")

        stats_results = validator.calculate_statistics(data, model_name)

        print(f"  Accuracy Range: {stats_results['min_accuracy']*100:.1f}% - {stats_results['max_accuracy']*100:.1f}% (Δ={stats_results['range']*100:.1f}%)")
        print(f"  Linear Trend: slope={stats_results['slope']:.6f}, p={stats_results['p_value']:.4f}, R²={stats_results['r_squared']:.4f}")

        if stats_results['p_value'] < 0.05:
            direction = "increasing" if stats_results['slope'] > 0 else "decreasing"
            print(f"  → Statistically significant {direction} trend (p < 0.05) ✓")
        else:
            print(f"  → No statistically significant trend (p >= 0.05)")

        print(f"  Early positions (mean): {stats_results['early_mean']*100:.1f}%")
        print(f"  Late positions (mean): {stats_results['late_mean']*100:.1f}%")

        if stats_results['t_pvalue'] is not None:
            print(f"  Early vs Late t-test: p={stats_results['t_pvalue']:.4f}")
            if stats_results['t_pvalue'] < 0.05:
                print(f"  → Statistically significant difference (p < 0.05) ✓")
            else:
                print(f"  → Not statistically significant (p >= 0.05)")

        print(f"  U-curve score: {stats_results['u_curve_score']:.4f}")
        if stats_results['u_curve_score'] > 0:
            print(f"  → Exhibits U-curve pattern (good at ends, bad in middle)")
        else:
            print(f"  → Exhibits recency bias (better at end than start)")

    # Final summary
    success = validator.print_summary()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
