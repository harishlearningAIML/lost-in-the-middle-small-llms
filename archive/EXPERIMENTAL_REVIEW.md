# Experimental Results Review: Lost in the Middle Study

**Review Date:** January 25, 2026  
**Experiment:** Testing "Lost in the Middle" phenomenon in small LLMs (2-4B parameters)

---

## Executive Summary

**Main Finding:** Small models (Gemma-2B, Gemma-4B, Llama-3B) show **recency bias** (better performance at end positions) rather than the U-curve pattern found in larger models (GPT-3.5, Claude) in the original "Lost in the Middle" paper.

**Statistical Status:** 
- ✅ Consistent pattern across all 3 models
- ⚠️ **Mixed statistical significance** - only 2/3 models show significant trends
- ⚠️ Early vs late comparisons not statistically significant (p > 0.05)

---

## 1. Experimental Design Review

### ✅ Strengths

1. **Appropriate Sample Size**
   - 30 trials per position (210 total per model)
   - Adequate for detecting moderate effect sizes
   - Follows standard practice for behavioral experiments

2. **Hard Distractors Design**
   - Uses same-entity distractors (e.g., multiple cities for "capital" question)
   - Creates realistic RAG scenario with semantic similarity
   - Prevents trivial "find the only relevant document" solutions

3. **Reproducibility**
   - Seeded random shuffling ensures reproducibility
   - Deterministic generation (temperature=0.0)
   - Clear documentation of methodology

4. **Multiple Positions**
   - 7 positions tested (1, 10, 25, 50, 75, 90, 100)
   - Good coverage from start to end
   - Includes "deep middle" positions

5. **Multiple Models**
   - Tests 3 different architectures (Gemma-2B, Gemma-4B, Llama-3B)
   - Allows for cross-model validation
   - Different context lengths (70 vs 100 docs) appropriately handled

### ⚠️ Limitations

1. **Limited Question Diversity**
   - Only 30 QA pairs total
   - All questions use same structure (factual retrieval)
   - May not generalize to other question types (reasoning, multi-hop)

2. **Position Selection**
   - Positions not evenly spaced (gaps: 9, 15, 25, 25, 15, 10)
   - Denser sampling at ends than middle
   - Could miss subtle middle-position effects

3. **Context Length Variation**
   - Llama uses 70 docs, Gemma uses 100 docs
   - Makes cross-model comparison less direct
   - However, this is justified by model limitations

4. **No Baseline Control**
   - No condition with gold document alone (no distractors)
   - Can't measure absolute performance degradation
   - Only relative position effects measured

---

## 2. Statistical Analysis Results

### Model-by-Model Analysis

#### **Gemma-2B (100 docs)**
- **Pattern:** Clear recency bias
- **Trend:** Increasing (slope = 0.000728, R² = 0.650)
- **Statistical Significance:** ✅ **p = 0.0285** (significant)
- **Early vs Late:** +7.2% improvement (p = 0.1625, not significant)
- **Accuracy Range:** 83.3% - 93.3%
- **Assessment:** Strong evidence for recency bias

#### **Gemma-4B (100 docs)**
- **Pattern:** Strong recency bias
- **Trend:** Increasing (slope = 0.001047, R² = 0.672)
- **Statistical Significance:** ✅ **p = 0.0240** (significant)
- **Early vs Late:** +9.4% improvement (p = 0.0519, marginal)
- **Accuracy Range:** 83.3% - 96.7%
- **Assessment:** Strongest evidence for recency bias

#### **Llama-3B (70 docs)**
- **Pattern:** Weak recency bias
- **Trend:** Increasing (slope = 0.000239, R² = 0.037)
- **Statistical Significance:** ❌ **p = 0.6782** (not significant)
- **Early vs Late:** +2.2% improvement (p = 0.5560, not significant)
- **Accuracy Range:** 90.0% - 100.0%
- **Assessment:** Pattern present but not statistically significant

### Key Statistical Findings

1. **Trend Consistency:** ✅ All 3 models show positive slope (recency bias)
2. **Significance:** ⚠️ Only 2/3 models show statistically significant trends
3. **Effect Size:** Moderate (7-9% improvement for Gemma models)
4. **U-Curve Rejection:** ✅ All models show negative U-curve scores (opposite of expected)

---

## 3. Interpretation Review

### ✅ Correct Interpretations

1. **"No U-Curve Pattern"** - ✅ **CORRECT**
   - All models show negative U-curve scores
   - Middle positions perform as well or better than early positions
   - Strongly contradicts original paper's findings

2. **"Recency Bias Present"** - ✅ **PARTIALLY CORRECT**
   - Pattern is consistent across models
   - However, statistical significance is mixed
   - Effect is moderate (7-9% for Gemma, 2% for Llama)

3. **"Early Positions Are Worst"** - ✅ **CORRECT**
   - Position 1 and 10 consistently show lowest accuracy
   - Contradicts primacy effect seen in larger models

### ⚠️ Overstated Claims

1. **"Strong Recency Bias"** - ⚠️ **OVERSTATED**
   - Only 2/3 models show significant trends
   - Llama-3B shows weak, non-significant effect
   - Should say "moderate recency bias" or "trend toward recency bias"

2. **"All Models Show Better Performance at End"** - ⚠️ **MOSTLY TRUE**
   - True for Gemma models (significant)
   - Llama shows pattern but not significant
   - Should qualify: "trend toward better performance"

3. **Early vs Late Comparisons** - ⚠️ **NOT STATISTICALLY SIGNIFICANT**
   - README claims "+6.7%", "+10.0%", "+1.7%" improvements
   - None of these are statistically significant (p > 0.05)
   - Should report p-values and note non-significance

---

## 4. Comparison to Original Paper

### Original "Lost in the Middle" Findings (Liu et al., 2023)

- **Models:** GPT-3.5, Claude (much larger models)
- **Pattern:** U-curve (good at start, bad in middle, good at end)
- **Effect:** Strong degradation in middle positions
- **Interpretation:** Attention mechanisms favor early and late tokens

### This Study's Findings

- **Models:** Gemma-2B, Gemma-4B, Llama-3B (small models)
- **Pattern:** Recency bias (worse at start, better at end)
- **Effect:** Moderate improvement at end positions
- **Interpretation:** Small models may have different attention patterns

### Key Differences

1. **Model Size:** 2-4B vs 100B+ parameters
2. **Pattern:** Opposite (recency vs U-curve)
3. **Effect Strength:** Moderate vs strong
4. **Architecture:** Different model families

**Conclusion:** The contradiction is **plausible** - small models may genuinely behave differently than large models. However, the effect is weaker and less consistently significant than claimed.

---

## 5. Potential Confounds & Limitations

### ⚠️ Experimental Confounds

1. **Question Order Effects**
   - Same 30 questions used for all positions
   - Order effects could influence results
   - **Mitigation:** Questions shuffled per position with seed

2. **Document Length Variation**
   - Gold documents may vary in length
   - Longer documents might be easier to find
   - **Check Needed:** Analyze if document length correlates with position

3. **Answer Format Consistency**
   - Some answers are single words, others are phrases
   - Extraction logic might favor certain formats
   - **Mitigation:** Evaluator handles multiple formats

4. **Hard Distractor Quality**
   - Distractors may vary in "hardness"
   - Some questions might have easier distractors
   - **Check Needed:** Analyze per-question accuracy

### 🔍 Missing Analyses

1. **Per-Question Breakdown**
   - Which questions show strongest position effects?
   - Are some questions position-invariant?
   - Could reveal question-type dependencies

2. **Error Analysis**
   - What types of errors occur at different positions?
   - Are models selecting wrong distractors?
   - Could inform mechanism understanding

3. **Token-Level Analysis**
   - Where in the context do models attend?
   - Could use attention weights (if available)
   - Would explain why recency bias occurs

4. **Confidence/Probability Scores**
   - Do models show different confidence at different positions?
   - Could reveal uncertainty patterns

---

## 6. Sample Size Adequacy

### Current Design
- **30 trials per position** × 7 positions = 210 trials per model
- **Total:** 630 trials across 3 models

### Power Analysis

For detecting a 10% difference (early vs late):
- **Required sample:** ~60 per group (early/late)
- **Current sample:** 60 early (2 positions × 30), 90 late (3 positions × 30)
- **Status:** ✅ Adequate

For detecting linear trend:
- **Required sample:** ~30 per position (for moderate effect)
- **Current sample:** 30 per position
- **Status:** ✅ Adequate

**Conclusion:** Sample size is adequate for detecting moderate effects, but may be underpowered for smaller effects (like Llama's 2% improvement).

---

## 7. Recommendations

### 🔴 Critical Issues

1. **Report Statistical Significance**
   - README should include p-values for all claims
   - Note that early vs late comparisons are not significant
   - Qualify "strong" claims as "moderate" or "trend"

2. **Qualify Llama Results**
   - Llama-3B shows pattern but not significant
   - Should note: "trend toward recency bias" not "recency bias"
   - Consider: Is 70-doc context too short to see effect?

### 🟡 Important Improvements

3. **Add Error Analysis**
   - Analyze which questions show strongest effects
   - Identify question types that are position-sensitive
   - Could strengthen or refine conclusions

4. **Report Effect Sizes**
   - Include confidence intervals
   - Report Cohen's d or similar effect size metrics
   - Helps readers assess practical significance

5. **Cross-Validation**
   - Consider running additional trials for Llama
   - Or acknowledge that Llama shows weaker, non-significant effect
   - Could test with more positions for Llama

### 🟢 Nice-to-Have Enhancements

6. **Baseline Condition**
   - Add condition with gold document alone
   - Measure absolute performance vs relative position effects
   - Would strengthen interpretation

7. **More Question Types**
   - Test reasoning questions, multi-hop questions
   - See if recency bias generalizes
   - Would increase external validity

8. **Attention Analysis**
   - If possible, analyze attention weights
   - Could explain mechanism behind recency bias
   - Would be valuable for understanding

---

## 8. Overall Assessment

### ✅ Strengths

1. **Clear experimental design** with appropriate controls
2. **Consistent pattern** across multiple models
3. **Realistic RAG scenario** with hard distractors
4. **Good documentation** and reproducibility
5. **Interesting finding** that contradicts original paper

### ⚠️ Weaknesses

1. **Statistical significance is mixed** - only 2/3 models significant
2. **Effect sizes are moderate** - not as strong as claimed
3. **Early vs late comparisons not significant** - main claim lacks support
4. **Limited question diversity** - may not generalize
5. **Missing error analysis** - don't know why errors occur

### 📊 Final Verdict

**Scientific Validity: 7/10**
- Well-designed experiment
- Clear methodology
- Statistical analysis needs improvement
- Claims slightly overstated

**Contribution: 8/10**
- Interesting finding that contradicts original paper
- Relevant for RAG applications
- Could influence retrieval pipeline design
- Needs stronger statistical support

**Recommendation:** 
- ✅ **Publishable** with revisions
- ⚠️ **Qualify claims** about statistical significance
- ⚠️ **Add error analysis** to strengthen conclusions
- ⚠️ **Report p-values** for all comparisons

---

## 9. Specific Corrections Needed

### README Corrections

1. **Table: "Early vs Late Position Performance"**
   - Current: Shows improvements without p-values
   - Should add: "p = 0.16, p = 0.05, p = 0.56" (not significant)
   - Or: "Trend toward improvement (not statistically significant)"

2. **"Strong Recency Bias" Claims**
   - Current: "Strong recency bias"
   - Should be: "Moderate recency bias" or "Trend toward recency bias"
   - Note: Only 2/3 models show significant trends

3. **Llama Results**
   - Current: Shows +1.7% improvement
   - Should note: "Non-significant trend (p = 0.56)"

### Statistical Reporting

Add to README:
```
### Statistical Significance

- **Gemma-2B:** Significant recency trend (p = 0.0285)
- **Gemma-4B:** Significant recency trend (p = 0.0240)  
- **Llama-3B:** Non-significant trend (p = 0.6782)

Early vs late position comparisons show moderate improvements 
(7-9% for Gemma models) but are not statistically significant 
(p > 0.05), suggesting the effect may be smaller than initially 
apparent or requires larger sample sizes to detect reliably.
```

---

## 10. Conclusion

This is a **well-executed experiment** that makes an **interesting contribution** to understanding how small LLMs handle long contexts. The finding that small models show recency bias (rather than U-curve) is **plausible and potentially important** for RAG applications.

However, the **statistical evidence is weaker than claimed**. The main findings are:
- ✅ Consistent pattern across models
- ✅ Significant trends in 2/3 models
- ⚠️ Early vs late comparisons not significant
- ⚠️ Effect sizes are moderate, not strong

**Recommendation:** Revise claims to be more conservative, add statistical reporting, and consider additional analyses to strengthen conclusions.

**Overall Grade: B+** (Good experiment, needs statistical rigor improvements)
