# Quick Summary: Experimental Results Review

## Key Findings

### ✅ What's Supported by Data

1. **Consistent Pattern:** All 3 models show recency bias (better at end positions)
2. **No U-Curve:** All models show negative U-curve scores (opposite of original paper)
3. **Early Positions Worst:** Position 1 and 10 consistently show lowest accuracy
4. **Significant Trends:** Gemma-2B and Gemma-4B show statistically significant increasing trends

### ⚠️ What Needs Qualification

1. **Statistical Significance:** Only 2/3 models show significant trends (Llama-3B: p = 0.68)
2. **Early vs Late:** Improvements (7-9%) are NOT statistically significant (p > 0.05)
3. **Effect Size:** Moderate effects, not "strong" as claimed
4. **Llama Results:** Shows pattern but not significant - may need more data

## Statistical Summary Table

| Model | Trend | P-value | Significant? | Early vs Late | P-value | Significant? |
|-------|-------|---------|--------------|---------------|---------|--------------|
| Gemma-2B | Increasing | 0.0285 | ✅ Yes | +7.2% | 0.1625 | ❌ No |
| Gemma-4B | Increasing | 0.0240 | ✅ Yes | +9.4% | 0.0519 | ❌ No (marginal) |
| Llama-3B | Increasing | 0.6782 | ❌ No | +2.2% | 0.5560 | ❌ No |

## Accuracy by Position

### Gemma-2B (100 docs)
- Position 1: 86.7% (worst)
- Position 10: 83.3% (worst)
- Position 25: 90.0%
- Position 50: 90.0%
- Position 75: 93.3%
- Position 90: 90.0%
- Position 100: 93.3% (best)

**Pattern:** Clear upward trend, significant (p = 0.0285)

### Gemma-4B (100 docs)
- Position 1: 86.7% (worst)
- Position 10: 83.3% (worst)
- Position 25: 90.0%
- Position 50: 96.7%
- Position 75: 93.3%
- Position 90: 93.3%
- Position 100: 96.7% (best)

**Pattern:** Strong upward trend, significant (p = 0.0240)

### Llama-3B (70 docs)
- Position 1: 93.3%
- Position 10: 93.3%
- Position 25: 93.3%
- Position 35: 93.3%
- Position 50: 96.7%
- Position 60: 100.0% (best)
- Position 70: 90.0%

**Pattern:** Weak upward trend, NOT significant (p = 0.6782)

## Main Issues Found

### 1. Overstated Claims
- README claims "strong recency bias" but:
  - Only 2/3 models show significant trends
  - Early vs late comparisons not significant
  - Effect sizes are moderate (7-9%), not strong

### 2. Missing Statistical Reporting
- No p-values reported in README
- Early vs late comparisons presented as definitive
- Should note non-significance

### 3. Llama Results Need Qualification
- Pattern present but not significant
- May need more trials or different analysis
- Should not be presented as strong evidence

## Recommendations

### Immediate Actions
1. ✅ Add p-values to README claims
2. ✅ Qualify "strong" as "moderate" or "trend"
3. ✅ Note that early vs late comparisons are not significant
4. ✅ Separate Llama results (weaker evidence)

### Future Improvements
1. Run more trials for Llama to increase power
2. Add error analysis (which questions show effects?)
3. Analyze per-question patterns
4. Consider baseline condition (gold doc alone)

## Overall Assessment

**Scientific Quality:** 7/10
- Good experimental design
- Clear methodology
- Statistical reporting needs improvement

**Contribution:** 8/10
- Interesting finding
- Contradicts original paper
- Relevant for RAG applications

**Recommendation:** 
- Publishable with revisions
- Qualify statistical claims
- Add p-values and effect sizes
