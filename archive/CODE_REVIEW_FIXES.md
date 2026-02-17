# Code Review Fixes Applied

Ruthless review identified critical issues. Here's what was fixed.

## 1. Reproducibility: `hash()` → `hashlib`

**Problem:** Python's `hash()` is randomized across runs (Python 3.3+). Experiments were not reproducible.

**Fix:** `run_experiment.py` now uses `hashlib.sha256(...).hexdigest()` for deterministic seeding.

## 2. Evaluator Bugs

**Problem A:** Substring false positives. Response "1887 was bad, the real answer is 1342" matched gold "1887".

**Fix:** Added `_extract_explicit_answer()` to detect when the model explicitly gives a different answer. Reject when explicit answer contradicts gold.

**Problem B:** Aggressive "the" prefix stripping broke "The Answer" (proper noun) → "Answer".

**Fix:** Removed standalone "the" from the prefix list. Only strip multi-word phrases like "the answer is".

**Problem C:** Number matching too loose. "2.4 in 1990 but 5 million" matched gold "2.4 million".

**Fix:** When matching by numbers, reject if `len(extracted) > 50` and gold has fewer numbers than extracted (wrong context).

## 3. Config Portability

**Problem:** Hardcoded paths like `/Volumes/T9/models/...` only work on one machine.

**Fix:** Model paths now read from env vars: `LOST_IN_MIDDLE_GEMMA_2B_PATH`, etc. Fallback to defaults in `config.py`.

## 4. Confounded Experimental Design

**Problem:** Llama at 70 docs vs Gemma at 100 docs plotted on same x-axis. "Late positions (75+)" meant different things (Llama: pos 70 of 70; Gemma: pos 75–100 of 100).

**Fix:**
- Unified `config.py` with `MODEL_CONFIG` per model (positions, total_docs)
- `run_experiment.py` uses model-specific config
- `create_charts.py` normalizes positions to **% of context** (0–100) for fair comparison
- Early vs late now defined as ≤20% and ≥70% of context

## 5. Input Validation

**Problem:** No validation of qa_pairs, positions, or distractors. Could fail with cryptic errors.

**Fix:** Added `validate_data()` that checks:
- QA pairs have required fields (id, question, answer, gold_doc)
- Positions are in range 1..total_docs
- Enough distractors (generic + hard) for total_docs

## 6. Statistical Rigor

**Problem:** No confidence intervals, no p-values. Claims of "recency bias" from n=30 with no statistical test.

**Fix:** Added `statistical_analysis.py`:
- Wilson score 95% CI per position
- Chi-squared test for early vs late
- Reports p-value; explicitly states when result is NOT significant

**Reality check:** Gemma-2B early (85%) vs late (91.7%): p=0.39. Not significant. The mentor was right.

## 7. Test Quality

**Problem:** `test_abbreviation_to_full_name` had `assert is_correct or "3" in str(is_correct)` — accepts anything.

**Fix:** Now asserts `is_correct is True` and added `test_reject_explicit_wrong_answer` for the false-positive case.

## What Was NOT Fixed (By Design)

- **Sample size:** Still 30 per position. Fix: run 100+ for real significance.
- **Synthetic data:** Still 30 fictional QA pairs. Fix: use NaturalQuestions/TriviaQA for paper comparison.
- **Logging:** Still uses `print()`. Could add `logging` module.
- **Number-in-wrong-context:** The "2.4 million" vs "2.4 in 1990 but 5 million" case matches via word overlap. Fixing would require semantic parsing; left as known edge case.

## Summary

| Issue | Status |
|-------|--------|
| hash() non-reproducible | Fixed |
| Evaluator false positives | Fixed |
| Evaluator "the" stripping | Fixed |
| Hardcoded paths | Fixed (env vars) |
| Confounded Llama vs Gemma | Fixed (normalized %) |
| No input validation | Fixed |
| No statistical tests | Fixed (new script) |
| Weak test assertions | Fixed |
| n=30 too small | Documented, not fixed |
| Synthetic data only | Documented, not fixed |
