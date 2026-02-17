# Results Review: Feb 9, 2026 Experiments

## CRITICAL BUG FOUND

**Config says 100 trials, but only 30 were run.**

All three result files claim `trials_per_position: 100` in the config, but the actual results show `total: 30` for every position.

**Root cause:** Line 101 in `run_experiment.py`:
```python
effective_trials = min(trials_per_position, len(qa_pairs))
```

The code silently caps trials at the number of QA pairs (30), but saves the original config value (100). This is misleading.

**Impact:** 
- Config claims 100 trials but only 30 were run
- You can't claim "100 trials" when you only have 30
- This undermines any statistical claims
- The code should either:
  1. Warn when `trials_per_position > len(qa_pairs)`
  2. Update the saved config to reflect `effective_trials`
  3. Allow repeating questions with different seeds to reach 100 trials

---

## Results Summary

### Gemma-2B (100 docs, 7 positions)
| Position | Correct | Total | Accuracy | 95% CI |
|----------|---------|-------|----------|--------|
| 1        | 26      | 30    | 86.7%    | [70.3%, 94.7%] |
| 10       | 26      | 30    | 86.7%    | [70.3%, 94.7%] |
| 25       | 28      | 30    | 93.3%    | [78.7%, 98.2%] |
| 50       | 29      | 30    | 96.7%    | [83.3%, 99.4%] |
| 75       | 25      | 30    | 83.3%    | [66.4%, 92.7%] |
| 90       | 28      | 30    | 93.3%    | [78.7%, 98.2%] |
| 100      | 30      | 30    | 100.0%   | [88.6%, 100.0%] |

**Pattern:** No clear trend. Position 75 (83.3%) is WORSE than position 1 (86.7%). Position 50 (96.7%) is best, not position 100.

**Early (1,10) vs Late (90,100):** 52/60 (86.7%) vs 58/60 (96.7%)
- Chi-squared: 2.727, p=0.0986
- **NOT statistically significant** (p > 0.05)

### Gemma-4B (100 docs, 7 positions)
| Position | Correct | Total | Accuracy | 95% CI |
|----------|---------|-------|----------|--------|
| 1        | 27      | 30    | 90.0%    | [74.4%, 96.5%] |
| 10       | 29      | 30    | 96.7%    | [83.3%, 99.4%] |
| 25       | 27      | 30    | 90.0%    | [74.4%, 96.5%] |
| 50       | 28      | 30    | 93.3%    | [78.7%, 98.2%] |
| 75       | 28      | 30    | 93.3%    | [78.7%, 98.2%] |
| 90       | 30      | 30    | 100.0%   | [88.6%, 100.0%] |
| 100      | 30      | 30    | 100.0%   | [88.6%, 100.0%] |

**Pattern:** Position 10 (96.7%) is HIGHER than position 1 (90.0%). Late positions (90, 100) are perfect, but so is position 10.

**Early (1,10) vs Late (90,100):** 56/60 (93.3%) vs 60/60 (100.0%)
- Chi-squared: 2.143, p=0.143
- **NOT statistically significant** (p > 0.05)

### Llama-3B (70 docs, 7 positions)
| Position | Correct | Total | Accuracy | 95% CI |
|----------|---------|-------|----------|--------|
| 1        | 27      | 30    | 90.0%    | [74.4%, 96.5%] |
| 10       | 28      | 30    | 93.3%    | [78.7%, 98.2%] |
| 25       | 29      | 30    | 96.7%    | [83.3%, 99.4%] |
| 35       | 29      | 30    | 96.7%    | [83.3%, 99.4%] |
| 50       | 27      | 30    | 90.0%    | [74.4%, 96.5%] |
| 60       | 28      | 30    | 93.3%    | [78.7%, 98.2%] |
| 70       | 29      | 30    | 96.7%    | [83.3%, 99.4%] |

**Pattern:** Position 50 (90.0%) is WORSE than position 1 (90.0%). No clear recency bias - position 25/35 are best, position 50 dips, then 60/70 recover.

**Early (1,10) vs Late (60,70):** 55/60 (91.7%) vs 57/60 (95.0%)
- Chi-squared: 0.351, p=0.554
- **NOT statistically significant** (p > 0.05)

---

## Statistical Analysis

### Confidence Intervals
With n=30, all confidence intervals are **wide** (±10-15%). For example:
- 86.7% accuracy has CI [70.3%, 94.7%] - a 24.4% range
- 100% accuracy has CI [88.6%, 100%] - still 11.4% uncertainty

**This means:** A difference of 10% (e.g., 86.7% vs 96.7%) could be noise.

### Chi-Squared Tests
All three models show **p > 0.05** for early vs late comparison:
- Gemma-2B: p=0.0986 (borderline, but not significant)
- Gemma-4B: p=0.143 (not significant)
- Llama-3B: p=0.554 (clearly not significant)

**Conclusion:** None of the "recency bias" claims are statistically supported.

---

## Problems Identified

### 1. Config Mismatch
- Config claims 100 trials
- Actual results show 30 trials
- **Fix:** Either fix the code to respect `trials_per_position`, or update config to match reality

### 2. No Clear Pattern
- Gemma-2B: Position 75 (83.3%) is WORSE than position 1 (86.7%)
- Gemma-4B: Position 10 (96.7%) matches late positions
- Llama-3B: Position 50 (90.0%) dips below position 1 (90.0%)

**This contradicts "recency bias"** - if recency bias exists, late positions should consistently outperform early ones.

### 3. Sample Size Still Too Small
Even if you had 100 trials, n=100 per position is still marginal for detecting 5-10% differences. You'd need:
- n=200+ for 95% power to detect 10% difference
- n=400+ for 95% power to detect 5% difference

### 4. Position 75 Anomaly (Gemma-2B)
Position 75 shows 83.3% (25/30) - the LOWEST accuracy. This is:
- Lower than position 1 (86.7%)
- Lower than position 10 (86.7%)
- 13.4% lower than position 100 (100%)

**This suggests:** Either random noise, or there's something specific about position 75 that hurts performance. Either way, it contradicts "recency bias."

### 5. Perfect Scores at Late Positions
Gemma-4B shows 100% at positions 90 and 100. With n=30, this could be:
- Real effect (recency helps)
- Luck (30/30 is possible by chance even if true rate is 90%)
- Easier questions (if QA pairs aren't randomized properly)

**Need to check:** Are the same 30 questions used at every position? If so, position effects are confounded with question difficulty.

---

## What the Data Actually Shows

### Gemma-2B
- **No clear trend.** Accuracy ranges 83.3% to 100% with no obvious pattern.
- Position 50 (96.7%) is best, not position 100.
- Position 75 (83.3%) is worst, contradicting recency bias.

### Gemma-4B
- **Weak trend toward late positions** (90% → 100%), but:
  - Position 10 (96.7%) already matches late performance
  - Position 25 (90.0%) dips below position 1 (90.0%)
  - Not statistically significant (p=0.143)

### Llama-3B
- **No recency bias.** Position 50 (90.0%) is equal to position 1 (90.0%).
- Best positions are 25/35 (96.7%), not late positions.
- Position 70 (96.7%) matches position 25 (96.7%), not better.

---

## Recommendations

### Immediate Fixes
1. **Fix the config bug** - Code should respect `trials_per_position` or config should match reality
2. **Run proper statistical tests** - Use the `statistical_analysis.py` script before making claims
3. **Increase sample size** - Run 200+ trials per position for real significance

### Analysis Fixes
1. **Check question randomization** - Are the same 30 questions used at every position? If so, you're not measuring position effects independently.
2. **Investigate position 75 anomaly** - Why does Gemma-2B drop to 83.3% at position 75?
3. **Normalize positions** - Compare % of context (0%, 10%, 25%, 50%, 75%, 90%, 100%) not raw positions, especially for Llama (70 docs) vs Gemma (100 docs)

### Reporting Fixes
1. **Don't claim "recency bias"** - The data doesn't support it statistically
2. **Report confidence intervals** - Show the uncertainty in your measurements
3. **Report p-values** - Be honest when results aren't significant
4. **Fix config mismatch** - Don't claim 100 trials when you have 30

---

## Bottom Line

**Your results do NOT support "recency bias" claims:**

1. **Gemma-2B:** Position 75 is WORSE than position 1 (contradicts recency)
2. **Gemma-4B:** p=0.143 (not significant)
3. **Llama-3B:** p=0.554 (clearly not significant), position 50 equals position 1

**What you CAN say:**
- "Small models show variable performance across positions"
- "Late positions sometimes perform better, but not consistently or significantly"
- "With n=30, confidence intervals are wide (±10-15%)"

**What you CANNOT say:**
- "Small models show recency bias" (not proven)
- "Late positions consistently outperform early ones" (Gemma-2B position 75 contradicts this)
- "Results are statistically significant" (all p > 0.05)

---

## Next Steps

1. Fix the config/trials bug
2. Run 200+ trials per position
3. Use proper statistical tests
4. Re-evaluate claims based on actual significance

The code fixes from the previous review are good, but the experimental results still don't support your conclusions.
