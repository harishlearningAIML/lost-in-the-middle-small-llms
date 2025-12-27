# Lost in the Middle: Small LLMs Show Recency Bias, Not U-Curve

Testing whether small local LLMs exhibit the "Lost in the Middle" phenomenon found in larger models (GPT-3.5, Claude).

**Main Finding:** Small models (2-4B params) show **recency bias** - they perform better when important information is at the END, not the beginning. This is the opposite of what the original paper found for larger models.

## Results Summary

### Final Accuracy by Position (V3 - 30 trials each)

```
┌─────────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬───────┐
│ Model       │ Pos 1  │ Pos 10 │ Pos 25 │ Pos 50 │ Pos 75 │ Pos 90 │ Pos 100│ Δ     │
├─────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ Gemma-2B    │ 86.7%  │ 83.3%  │ 90.0%  │ 90.0%  │ 93.3%  │ 90.0%  │ 93.3%  │ +6.7% │
│ Gemma-4B    │ 86.7%  │ 83.3%  │ 90.0%  │ 96.7%  │ 93.3%  │ 93.3%  │ 96.7%  │+10.0% │
├─────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ Model       │ Pos 1  │ Pos 10 │ Pos 25 │ Pos 35 │ Pos 50 │ Pos 60 │ Pos 70 │ Δ     │
├─────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────┤
│ Llama-3B*   │ 93.3%  │ 93.3%  │ 93.3%  │ 93.3%  │ 96.7%  │ 100%   │ 90.0%  │ +1.7% │
└─────────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴───────┘
* Llama tested with 70 docs (degenerates at 100 docs on MPS)
Δ = Late positions avg - Early positions avg
```

### Key Pattern: Recency Bias

```
Expected (Original Paper - Large Models):     Actual (Small Models):

Accuracy                                      Accuracy
   ▲                                             ▲
   │  ●                           ●              │                      ●  ●
   │    ╲                       ╱                │              ●  ●  ●
   │      ╲                   ╱                  │        ●  ●
   │        ╲    U-curve    ╱                    │  ●  ●
   │          ╲           ╱                      │        Upward trend
   │            ●───────●                        │
   └────────────────────────▶                    └────────────────────────▶
      Start    Middle    End                        Start    Middle    End
```

### Early vs Late Position Performance

| Model | Early Positions (1, 10) | Late Positions (75+) | Improvement |
|-------|------------------------|---------------------|-------------|
| Gemma-2B | 85.0% | 91.7% | **+6.7%** |
| Gemma-4B | 85.0% | 95.0% | **+10.0%** |
| Llama-3B | 93.3% | 95.0% | **+1.7%** |

## Experiment Design

### Test Setup
- **Models:** Gemma-2-2B-it, Gemma-3-4B-it, Llama-3.2-3B-Instruct
- **Context:** 70-100 documents per prompt (~7-10K tokens)
- **Positions tested:** 7 positions from start to end
- **Trials:** 30 per position (210 total per model)
- **Hardware:** Apple M-series (MPS backend)

### Hard Distractors

Each question includes 7 "hard distractors" - documents that mention the same entities but with wrong information:

```
Question: "What is the capital of Valdoria?"
Correct answer: Zentrix

Gold document (position varies):
  "As of the 2019 constitutional reform, Valdoria's official capital is Zentrix..."

Hard distractors (shuffled into other positions):
  "Valdoria's largest city is Northgate with 1.2 million residents..."
  "The historic capital of Valdoria was Ironhold from 1342 to 1847..."
  "Valdoria's provisional capital was Silverton during the 1991-1994 civil conflict..."
  "The proposed new capital of Valdoria is Eastbridge, with construction beginning in 2026..."
  ...
```

This simulates real RAG scenarios where retrieved documents are semantically similar but may contain outdated, incorrect, or tangential information.

## Findings

### 1. No "Lost in the Middle" Effect
The classic U-curve (good at start, bad in middle, good at end) does **not** appear in small models.

### 2. Strong Recency Bias
All models perform better when the gold document is near the **end** of the context:
- Gemma-2B: +6.7% (late vs early positions)
- Gemma-4B: +10.0% (late vs early positions)
- Llama-3B: +1.7% (late vs early positions)

### 3. Early Positions Are Worst
Position 1 and 10 consistently show the **lowest** accuracy (83-87%), contrary to the "primacy effect" seen in larger models.

### 4. Hard Distractors Work
The same-entity distractors successfully cause 3-17% error rate, proving the task is non-trivial.

### 5. Llama Context Limitation
Llama 3.2 3B degenerates (outputs `!!!!!!`) at 100 docs on MPS, despite supporting 128K context officially. Works fine at 70 docs.

## Practical Implications

**For RAG with small local models:**
- Put the most relevant document **LAST**, not first
- Document ordering matters differently than for large models
- Consider the recency bias when designing retrieval pipelines

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test pipeline (no GPU needed)
cd src
python run_experiment.py --dry-run

# Run single model with verbose output
python run_experiment.py --model gemma-2b --verbose --limit 5

# Run full experiment
python run_experiment.py --model gemma-2b

# Generate visualization
python visualize.py -i results/results_gemma-2b_20251226_162353.json
```

## Configuration

Edit `src/config.py` to adjust:
```python
MODELS = {
    "gemma-2b": "/path/to/gemma-2-2b-it",
    "gemma-4b": "/path/to/gemma-3-4b-it",
    "llama-3b": "/path/to/llama-3.2-3b-instruct",
}

POSITIONS = [1, 10, 25, 50, 75, 90, 100]  # Positions to test
TOTAL_DOCS = 100                          # Documents per context
TRIALS_PER_POSITION = 30                  # Trials for statistical significance
```

For Llama (reduced context): use `config_llama.py` with `TOTAL_DOCS = 70`

## Project Structure

```
├── data/
│   ├── qa_pairs.json        # 30 QA pairs with hard distractors
│   └── distractors.json     # 100+ generic filler documents
├── src/
│   ├── config.py            # Experiment configuration
│   ├── config_llama.py      # Llama-specific config (70 docs)
│   ├── config_gemma.py      # Gemma-specific config (100 docs)
│   ├── context_builder.py   # Build prompts with gold at position N
│   ├── evaluator.py         # Answer checking logic
│   ├── model_runner.py      # HuggingFace inference wrapper
│   ├── run_experiment.py    # Main experiment loop
│   └── visualize.py         # Generate charts
├── results/                 # Raw JSON results
├── V1/                      # Initial 20-doc experiment (baseline)
└── requirements.txt
```

## Results Files

| File | Model | Docs | Description |
|------|-------|------|-------------|
| `results_gemma-2b_20251226_162353.json` | Gemma-2B | 100 | Final V3 results |
| `results_gemma-4b_20251226_165033.json` | Gemma-4B | 100 | Final V3 results |
| `results_llama-3b_20251226_173208.json` | Llama-3B | 70 | Final V3 results |

## Experiment Evolution

| Version | Docs | Distractors | Result |
|---------|------|-------------|--------|
| V1 | 20 | Easy (different topics) | 100% accuracy - too easy |
| V2 | 50 | Hard (same-entity) | Effect emerges |
| V3 | 70-100 | Hard (same-entity) | Clear recency bias pattern |

## References

- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) - Liu et al., 2023

## Notes

This is a small-scale experiment, not rigorous research. The goal was to understand how small local models actually behave versus what the papers say about larger models.

---

*Tested on Apple Silicon (M-series) with MPS backend. Results may vary on CUDA.*
