# Lost in the Middle Experiment - V2 (Harder)

Testing whether small LLMs exhibit the "Lost in the Middle" phenomenon where they disproportionately ignore information in the middle of long contexts.

## Changes from V1

| Aspect | V1 | V2 |
|--------|----|----|
| Documents per context | 20 | **50** |
| Positions tested | 1, 5, 10, 15, 20 | **1, 10, 25, 40, 50** |
| Distractors | Generic (different topics) | **Hard (same-entity confusion)** |
| Example distractor | "Marvania's capital is..." | "Valdoria's former capital was..." |

## Why V2 Should Work

The original paper found effects at 4K+ tokens. V2 changes:

1. **More documents** (50 vs 20) = longer context ≈ 5-6K tokens
2. **Confusing distractors** - each question has 5 same-entity distractors:
   - Q: "What is the capital of Valdoria?"
   - Hard distractor: "Valdoria's largest city is Northgate..." (plausible but wrong)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test pipeline (no GPU)
cd src
python run_experiment.py --dry-run

# Run single model with verbose output (see each response)
python run_experiment.py --model gemma-2b --verbose --limit 5

# Run full experiment
python run_experiment.py --model gemma-2b

# Run all models
python run_experiment.py

# Generate charts
python visualize.py
```

## Command Line Options

```
--model MODEL     Run specific model only (gemma-2b, gemma-4b, llama-3b)
--dry-run         Test without actual inference
--verbose, -v     Print each question/response
--limit N, -l N   Limit to N trials per position
--output FILE     Output path for results JSON
```

## Expected Results

With harder distractors, you should see accuracy degradation:

```
Model           | Pos  1 | Pos 10 | Pos 25 | Pos 40 | Pos 50 | Drop
--------------------------------------------------------------------
gemma-2b        |  85.0% |  70.0% |  55.0% |  65.0% |  80.0% | 30.0%
gemma-4b        |  90.0% |  75.0% |  60.0% |  72.0% |  85.0% | 30.0%
llama-3b        |  88.0% |  73.0% |  58.0% |  70.0% |  82.0% | 30.0%
```

The U-shaped curve shows:
- **High accuracy** at position 1 (recency bias in prompt)
- **Low accuracy** at position 25 (middle)
- **Recovering accuracy** at position 50 (primacy bias)

## Project Structure

```
lost-in-middle-v2/
├── data/
│   ├── qa_pairs.json      # 20 QA pairs with hard distractors
│   └── distractors.json   # 50 generic distractors
├── src/
│   ├── config.py          # Model paths and experiment settings
│   ├── context_builder.py # Build prompts with gold at position N
│   ├── evaluator.py       # Check if answer is correct
│   ├── model_runner.py    # HuggingFace inference
│   ├── run_experiment.py  # Main experiment loop
│   └── visualize.py       # Generate charts
├── results/
│   └── results_v2.json    # Raw results
├── requirements.txt
└── README.md
```

## Troubleshooting

**Still 100% accuracy?**
- Context might still be too short for your models
- Try increasing `TOTAL_DOCS` to 100 in `config.py`
- Add more hard distractors per question

**Out of memory?**
- Reduce `TOTAL_DOCS` in `config.py`
- Use `--limit 5` to test fewer questions first

**Flat results (no curve)?**
- Model might handle this context length well
- Try with even smaller models
- Increase context length further

## References

- [Lost in the Middle paper](https://arxiv.org/abs/2307.03172) - Liu et al., 2023
