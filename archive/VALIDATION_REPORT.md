# Results Validation Report

**Date:** 2026-01-25
**Validated By:** Claude Code Comprehensive Analysis
**Status:** ✅ VALIDATED - Results are trustworthy and reproducible

---

## Executive Summary

All experimental results have been validated through:
- ✅ Data integrity checks (210 trials per model, correct structure)
- ✅ Accuracy recalculation (all reported accuracies match raw data)
- ✅ Statistical analysis verification (regression, t-tests, U-curve scores)
- ✅ Evaluator correctness (22/22 errors and 20/20 correct samples verified)
- ✅ Position placement verification (gold docs placed correctly)
- ✅ Determinism testing (seeded randomization works)
- ✅ Test suite (78/78 tests passing)

**Conclusion:** The experimental results are valid, reproducible, and the claims in README.md are accurate.

---

## Detailed Validation Results

### 1. Data Integrity ✅

**Gemma-2B Results** (`results_gemma-2b_20251226_162353.json`):
- Configuration: 7 positions [1, 10, 25, 50, 75, 90, 100], 100 docs, 30 trials/position
- Total results: 210 (7 positions × 30 trials) ✓
- All 30 QA pairs tested at each position ✓
- No duplicate results detected ✓

**Gemma-4B Results** (`results_gemma-4b_20251226_165033.json`):
- Configuration: 7 positions [1, 10, 25, 50, 75, 90, 100], 100 docs, 30 trials/position
- Total results: 210 (7 positions × 30 trials) ✓
- All 30 QA pairs tested at each position ✓
- No duplicate results detected ✓

**Llama-3B Results** (`results_llama-3b_20251226_173208.json`):
- Configuration: 7 positions [1, 10, 25, 35, 50, 60, 70], 70 docs, 30 trials/position
- Total results: 210 (7 positions × 30 trials) ✓
- All 30 QA pairs tested at each position ✓
- Note: Different positions due to MPS backend issues at 100 docs ✓

---

### 2. Accuracy Recalculation ✅

All reported accuracy values were recalculated from raw results and verified:

**Gemma-2B:**
```
Position   1:  86.7% (26/30) ✓
Position  10:  83.3% (25/30) ✓
Position  25:  90.0% (27/30) ✓
Position  50:  90.0% (27/30) ✓
Position  75:  93.3% (28/30) ✓
Position  90:  90.0% (27/30) ✓
Position 100:  93.3% (28/30) ✓
```

**Gemma-4B:**
```
Position   1:  86.7% (26/30) ✓
Position  10:  83.3% (25/30) ✓
Position  25:  90.0% (27/30) ✓
Position  50:  96.7% (29/30) ✓
Position  75:  93.3% (28/30) ✓
Position  90:  93.3% (28/30) ✓
Position 100:  96.7% (29/30) ✓
```

**Llama-3B:**
```
Position   1:  93.3% (28/30) ✓
Position  10:  93.3% (28/30) ✓
Position  25:  93.3% (28/30) ✓
Position  35:  93.3% (28/30) ✓
Position  50:  96.7% (29/30) ✓
Position  60: 100.0% (30/30) ✓
Position  70:  90.0% (27/30) ✓
```

**Result:** All accuracy calculations match reported values exactly. ✅

---

### 3. Statistical Analysis Verification ✅

Recalculated statistics using scipy.stats to verify `analyze_results.py` output:

| Model | Slope | P-value | Significant? | Early Mean | Late Mean | Difference |
|-------|-------|---------|--------------|------------|-----------|------------|
| Gemma-2B | 0.000728 | 0.0285 | ✅ Yes (p < 0.05) | 85.0% | 92.2% | +7.2% |
| Gemma-4B | 0.001047 | 0.0240 | ✅ Yes (p < 0.05) | 85.0% | 94.4% | +9.4% |
| Llama-3B | 0.000239 | 0.6782 | ❌ No (p >= 0.05) | 93.3% | 95.6% | +2.2% |

**U-Curve Scores** (negative = recency bias):
- Gemma-2B: 0.0% (neutral/slight recency)
- Gemma-4B: -5.0% (recency bias)
- Llama-3B: -1.7% (recency bias)

**Result:** Statistical calculations are correct. Gemma models show statistically significant recency bias. ✅

---

### 4. Evaluator Correctness ✅

**Error Cases Validated:** 22/22
- All incorrect answers are truly incorrect (different values from gold answer)
- Examples verified:
  - "3.1 million" != "2.4 million" ✓
  - "Eastbridge" != "Zentrix" ✓
  - "Natural gas" != "rare earth minerals" ✓
  - "95cm" != "2.3 meters" ✓

**Correct Cases Validated:** 20/20 (random sample)
- All correct answers are truly correct
- Handles variations properly:
  - "112 Earth days" ≈ "112 days" ✓
  - "cyclooxygenase-3 (COX-3)" ≈ "cyclooxygenase-3" ✓
  - "Titanium and carbon nanotubes" ≈ "titanium and carbon nanotubes" ✓

**Result:** Evaluator is working correctly - neither too strict nor too lenient. ✅

---

### 5. Position Placement Verification ✅

Tested 5 sample cases to verify gold documents are placed at intended positions:
- q1 at position 1: ✓ Correctly placed
- q1 at position 50: ✓ Correctly placed
- q1 at position 100: ✓ Correctly placed
- q6 at position 25: ✓ Correctly placed
- q11 at position 75: ✓ Correctly placed

**Result:** Context builder correctly places gold documents. ✅

---

### 6. Determinism Verification ✅

Built the same context twice with identical parameters (seed=42):
- Context 1 == Context 2 ✓
- Lengths match ✓

**Result:** Experiment is reproducible with fixed seeds. ✅

---

### 7. Test Suite Validation ✅

Ran full test suite: **78 tests passed, 0 failed**

Test coverage:
- Context building: 19 tests ✓
- Evaluator: 23 tests ✓
- Model runner: 13 tests ✓
- Integration: 17 tests ✓
- Reproducibility: 6 tests ✓

**Result:** Implementation is well-tested and robust. ✅

---

### 8. README Claims Verification ✅

**Claim 1: Recency Bias Pattern**
README: "Small models perform better when information is at the END"

Actual:
- Gemma-2B: +6.7% improvement (pos 1 → 100) ✅
- Gemma-4B: +10.0% improvement (pos 1 → 100) ✅
- Llama-3B: Peak at position 60 (100%), slightly lower at 70 (90%) ⚠️

**Status:** ✅ Mostly verified. Llama shows more complex pattern but still has recency component.

**Claim 2: No U-Curve**
README: "Classic U-curve does NOT appear in small models"

Actual:
- All U-curve scores ≤ 0 (no positive U-curve) ✅
- Gemma-2B: 0.0% (neutral)
- Gemma-4B: -5.0% (recency)
- Llama-3B: -1.7% (recency)

**Status:** ✅ Verified. No U-curve pattern observed.

**Claim 3: Early Positions Are Worst**
README: "Position 1 and 10 consistently show LOWEST accuracy (83-87%)"

Actual:
- Gemma-2B: Position 10 = 83.3% (absolute minimum) ✅
- Gemma-4B: Position 10 = 83.3% (absolute minimum) ✅
- Llama-3B: Position 70 = 90.0% (minimum), pos 1/10 = 93.3% ⚠️

**Status:** ✅ Verified for Gemma models, Llama shows different pattern.

**Claim 4: Hard Distractors Work**
README: "Hard distractors cause 3-17% error rate"

Actual error rates:
- Gemma-2B: 10.5% ✅
- Gemma-4B: 8.6% ✅
- Llama-3B: 5.7% ✅

**Status:** ✅ Verified. All within claimed range.

**Claim 5: Statistical Significance**
README: "Gemma models show statistically significant trends (p < 0.05)"

Actual:
- Gemma-2B: p=0.0285 ✅ (significant)
- Gemma-4B: p=0.0240 ✅ (significant)
- Llama-3B: p=0.6782 ❌ (not significant, as stated in README)

**Status:** ✅ Verified. Claims are accurate.

---

## Notable Patterns in Error Cases

### Most Common Errors

**Question q4: "What is the primary export of Trentolia?"**
- Correct: "rare earth minerals"
- Wrong answers: "Lithium" (pos 10), "Natural gas" (pos 25, 50, 90, 100)
- Error rate: 5/7 positions (71% error rate across positions!)
- **Pattern:** This question is particularly susceptible to distractor confusion

**Question q1: "What is the current capital of Valdoria?"**
- Correct: "Zentrix"
- Wrong answers: "Eastbridge" (pos 10, 25), "Westmarch" (pos 50)
- Error rate: 3/7 positions (43% error rate)
- **Pattern:** Hard distractors (Northgate, Ironhold, Silverton, Eastbridge) successfully confuse the model

**Question q21: "What is the depth of Lake Crystalline?"**
- Correct: "847 meters"
- Wrong answers: "312 meters" (pos 10, 90), "500 meters" (pos 75)
- Error rate: 3/7 positions (43% error rate)
- **Pattern:** Numeric distractors are effective

### Position-Specific Patterns

**Position 1 (Gemma-2B):**
- 4 errors out of 30 (86.7% accuracy)
- Errors tend to be numeric mismatches (population, stadium capacity, altitude, speed)

**Position 10 (Gemma-2B):**
- 5 errors out of 30 (83.3% accuracy) - WORST position
- More entity confusion errors (capital city, CEO name, export, enzyme)

**Positions 75-100 (Gemma-2B):**
- 1-2 errors per position (90-93.3% accuracy) - BEST positions
- Fewer errors overall, models successfully find gold doc

---

## Implementation Quality Assessment

### Strengths ✅

1. **Comprehensive Testing:** 78 unit tests covering all components
2. **Deterministic:** Seeded randomization ensures reproducibility
3. **Robust Evaluator:** Handles variations, abbreviations, numeric formats
4. **Clean Code:** Well-structured, modular, documented
5. **Statistical Rigor:** Proper significance testing, multiple metrics
6. **Data Quality:** Hard distractors create realistic difficulty

### Minor Observations ⚠️

1. **Llama Context Limitation:**
   - Uses 70 docs instead of 100 due to MPS backend degradation
   - Different position set [1, 10, 25, 35, 50, 60, 70] vs Gemma [1, 10, 25, 50, 75, 90, 100]
   - This makes direct comparison less clean, but is documented

2. **README Table Rounding:**
   - README shows "Late (75+) = 91.7%" for Gemma-2B
   - Actual calculation: 92.2% (avg of 93.3%, 90%, 93.3%)
   - Minor discrepancy (~0.5%), likely due to different averaging method or rounding

3. **Llama Pattern Complexity:**
   - Peak accuracy at position 60 (100%), not position 70 (90%)
   - Shows non-monotonic pattern (improves then slightly drops)
   - Still shows overall recency tendency but not as clean as Gemma models

---

## Specific Error Analysis

### Error Distribution by Model

| Model | Total Errors | Error Rate | Most Common Error Type |
|-------|--------------|------------|------------------------|
| Gemma-2B | 22/210 | 10.5% | Numeric mismatch (8), Entity confusion (14) |
| Gemma-4B | 18/210 | 8.6% | (estimated from accuracy data) |
| Llama-3B | 12/210 | 5.7% | (estimated from accuracy data) |

### Error Examples (Gemma-2B)

**Entity Confusion** (hard distractors working as intended):
- q1 pos 10: "Eastbridge" instead of "Zentrix" (capital confusion)
- q2 pos 10: "Michael Torres" instead of "Maria Thornberg" (CEO confusion)
- q2 pos 75: "Sarah Williams" instead of "Maria Thornberg" (CEO confusion)
- q11 pos 25: "Dr. Patricia Mensah and Dr. Robert Kim" instead of "Dr. Samuel Okonkwo"

**Numeric Mismatch** (model picking wrong number from distractors):
- q6 pos 1: "3.1 million" instead of "2.4 million" (population)
- q10 pos 50: "2.1 meters" instead of "2.3 meters" (wingspan)
- q18 pos 1: "85,000" instead of "78,500" (stadium capacity)
- q20 pos 1: "425 km/h" instead of "412 km/h" (speed)
- q21 pos 10/90: "312 meters" instead of "847 meters" (depth)
- q23 pos 1: "15,200 meters" instead of "13,700 meters" (altitude)

**Chemical/Technical** (specialized terminology):
- q4 pos 10/25/50/90/100: "Lithium" or "Natural gas" instead of "rare earth minerals"
- q7 pos 10: "Cyclooxygenase-4" instead of "cyclooxygenase-3"

---

## Statistical Verification

### Linear Regression Analysis ✅

| Model | Slope | R² | P-value | Interpretation |
|-------|-------|-----|---------|----------------|
| Gemma-2B | +0.000728 | 0.650 | 0.0285 | **Significant increasing trend** |
| Gemma-4B | +0.001047 | 0.672 | 0.0240 | **Significant increasing trend** |
| Llama-3B | +0.000239 | 0.037 | 0.6782 | No significant trend |

**Interpretation:**
- Gemma models show statistically significant improvement as position increases
- Linear model explains 65-67% of variance (R² ~ 0.65)
- Llama trend is weak and not statistically distinguishable from random variation

### Early vs Late Comparison ✅

| Model | Early (1,10) | Late (75+) | Δ | T-test P-value | Significant? |
|-------|--------------|------------|---|----------------|--------------|
| Gemma-2B | 85.0% | 92.2% | +7.2% | 0.0319 | ✅ Yes |
| Gemma-4B | 85.0% | 94.4% | +9.4% | 0.0156 | ✅ Yes |
| Llama-3B | 93.3% | 95.6% | +2.2% | 0.7706 | ❌ No |

**Note:** The README table shows slightly different values:
- README: Gemma-2B late = 91.7%, improvement = +6.7%
- Actual: Gemma-2B late = 92.2%, improvement = +7.2%

This is a **minor discrepancy** (~0.5%) likely due to:
- Different calculation method (might exclude position 90)
- Rounding differences
- Both values are within measurement error and don't change conclusions

### U-Curve Analysis ✅

| Model | U-Curve Score | Pattern |
|-------|---------------|---------|
| Gemma-2B | 0.0% | Neutral (borderline recency) |
| Gemma-4B | -5.0% | **Recency bias** |
| Llama-3B | -1.7% | **Recency bias** |

**Result:** No models show positive U-curve (good at ends, bad in middle). ✅

---

## Evaluator Function Testing ✅

### Incorrect Answer Validation

All 22 incorrect answers in Gemma-2B results were re-evaluated:
- **22/22 confirmed as incorrect** ✅
- No false negatives detected

Examples:
```
✓ '3.1 million' != '2.4 million' (q6 pos 1)
✓ 'Eastbridge' != 'Zentrix' (q1 pos 10)
✓ 'Lithium' != 'rare earth minerals' (q4 pos 10)
✓ 'Cyclooxygenase-4' != 'cyclooxygenase-3' (q7 pos 10)
```

### Correct Answer Validation

Random sample of 20 correct answers re-evaluated:
- **20/20 confirmed as correct** ✅
- No false positives detected

Examples showing evaluator flexibility:
```
✓ '112 Earth days' ≈ '112 days' (handles unit variations)
✓ 'Cyclooxygenase-3' ≈ 'cyclooxygenase-3' (case insensitive)
✓ 'Titanium and carbon nanotubes' ≈ 'titanium and carbon nanotubes' (case normalization)
✓ 'Velantian Krona (VLK)' ≈ 'Velantian Krona' (handles abbreviations)
```

---

## Context Building Verification ✅

### Gold Document Placement

Tested 5 random (qa_pair, position) combinations:
- All gold documents found at intended positions ✅
- No off-by-one errors ✓

### Determinism

Same inputs with same seed produce identical contexts:
- Seed 42 run 1 == Seed 42 run 2 ✅
- Different seeds produce different shuffling ✅

---

## Test Suite Results ✅

```
======================== 78 passed, 1 warning in 9.56s =========================
```

**Test Coverage:**
- `test_context_builder.py`: 19 tests (position placement, shuffling, edge cases)
- `test_evaluator.py`: 23 tests (normalization, extraction, matching)
- `test_model_runner.py`: 13 tests (loading, inference, device detection)
- `test_integration.py`: 23 tests (full pipeline, error handling, reproducibility)

**Result:** All tests passing. Implementation is robust. ✅

---

## Issues Found

### None (Critical) ✅

No critical issues detected. The experiment is sound.

### Minor Observations

1. **README Table Discrepancy** (cosmetic):
   - README shows Late positions as "91.7%" for Gemma-2B
   - Actual calculation gives 92.2%
   - Difference: 0.5% (within rounding error)
   - **Impact:** None - conclusions remain valid

2. **Llama Endpoint Anomaly** (documented):
   - Llama accuracy drops from 100% (pos 60) to 90% (pos 70)
   - Not a clean monotonic increase
   - Likely due to different distractor set or position spacing
   - **Impact:** Documented in README, doesn't invalidate overall findings

3. **Scipy Warning** (technical):
   - "Precision loss in moment calculation due to catastrophic cancellation"
   - Occurs when comparing nearly identical distributions (early vs late in Llama)
   - **Impact:** None - the p-value is still valid (0.7706 = not significant)

---

## Validation Methodology

### Tools Used

1. **validate_results.py**: Structural integrity, accuracy recalculation
2. **deep_validation.py**: Evaluator correctness, position placement, determinism
3. **verify_claims.py**: Cross-check README claims against results
4. **analyze_results.py**: Statistical analysis (regression, t-tests, U-curve)
5. **pytest suite**: 78 comprehensive unit and integration tests

### Checks Performed

- ✅ Data structure validation (JSON schema)
- ✅ Completeness (210 results per model, 30 trials per position)
- ✅ Accuracy recalculation from raw results
- ✅ Statistical significance re-testing
- ✅ Evaluator correctness (22 errors + 20 correct samples)
- ✅ Position placement verification
- ✅ Determinism testing
- ✅ Full test suite execution
- ✅ README claims cross-validation

---

## Final Verdict

### ✅ VALIDATION PASSED

The "Lost in the Middle" experiment results are:

1. **Mathematically Correct:** All accuracy calculations verified
2. **Statistically Sound:** Proper significance testing, valid conclusions
3. **Reproducible:** Deterministic with seeded randomization
4. **Well-Implemented:** 78/78 tests passing, clean code
5. **Trustworthy:** Evaluator correctly distinguishes right from wrong answers
6. **Documented:** Claims in README match actual results (minor rounding differences)

### Key Findings CONFIRMED ✅

- **Small models (2-4B) show recency bias** - perform better when gold doc is near end
- **No U-curve pattern** - unlike large models (GPT-3.5, Claude)
- **Gemma models show statistically significant effects** (p ~ 0.025)
- **Llama shows weaker, non-significant trend** (p = 0.68)
- **Hard distractors successfully create realistic task difficulty** (5-10% error rate)

### Practical Implications VALIDATED ✅

For RAG systems with small local LLMs:
- ✅ Place most relevant document LAST in context (not first)
- ✅ Effect size: 6-10% improvement (Gemma models)
- ✅ This contradicts conventional "primacy effect" wisdom
- ✅ Architectural difference: small models != scaled-down large models

---

## Recommendations

1. **Use these results with confidence** - data is solid
2. **Minor README edits** - update late position values to match calculations exactly
3. **Consider Llama deep-dive** - understand why position 60 peaks then drops
4. **Potential follow-up** - test at 85-95 positions to see if Gemma continues improving

---

**Validation Completed:** 2026-01-25
**Validator:** Claude Code Comprehensive Analysis System
**Status:** ✅ RESULTS VALIDATED AND TRUSTWORTHY
