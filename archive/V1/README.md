# Lost in the Middle: Context Position Bias Testing

Testing whether small open-source LLMs suffer from the "lost in the middle" phenomenon documented in [Stanford's 2023 paper](https://arxiv.org/abs/2307.03172).

## Thesis

> Models claim large context windows but disproportionately ignore information in the middle positions. I tested this on small open source models (Gemma-2B, Gemma-4B, Llama-3.2-3B).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run dry-run test (no GPU needed)
python run_experiment.py --dry-run

# Run single model
python run_experiment.py --model gemma-2b

# Run all models
python run_experiment.py

# Generate visualizations
python visualize.py
```

## Project Structure

```
lost-in-middle/
├── config.py              # Experiment configuration
├── data/
│   ├── qa_pairs.json      # 20 QA pairs with fake entities
│   └── distractors.json   # 30 distractor documents
├── context_builder.py     # Build prompts with gold at position N
├── evaluator.py           # Check if model answer is correct
├── model_runner.py        # Load and run HuggingFace models
├── run_experiment.py      # Main experiment loop
├── visualize.py           # Generate charts
└── results/               # Output directory
    ├── results.json       # Raw results
    ├── position_accuracy.png  # U-shaped curve chart
    ├── heatmap.png        # Model × Position heatmap
    └── summary_stats.json # Summary statistics
```

## Experiment Design

### Setup
- **Models**: Gemma-2B, Gemma-4B, Llama-3.2-3B
- **Context**: 20 documents (1 gold + 19 distractors)
- **Positions tested**: 1, 5, 10, 15, 20
- **Trials per position**: 20 different QA pairs

### Why Fake Entities?
All QA pairs use fictional entities (countries, companies, people) to avoid training data contamination. The model can't "know" the answer - it must find it in context.

### Evaluation
Simple string matching with variants. If the model's answer contains the gold answer (or any acceptable variant), it's correct.

## Expected Results

```
Position  | Gemma-2B | Gemma-4B | Llama-3B
----------|----------|----------|----------
1 (start) |   ~75%   |   ~80%   |   ~78%
5         |   ~65%   |   ~70%   |   ~68%
10 (mid)  |   ~45%   |   ~55%   |   ~50%
15        |   ~60%   |   ~65%   |   ~62%
20 (end)  |   ~70%   |   ~75%   |   ~72%
```

Classic U-shaped curve: high accuracy at start/end, drop in the middle.

## Key Metrics

1. **Accuracy by Position**: Does middle drop?
2. **Drop Magnitude**: How much worse is middle vs. best?
3. **Model Comparison**: Do larger models handle it better?

## Extending

### Add More Positions
Edit `config.py`:
```python
POSITIONS = [1, 3, 5, 7, 10, 13, 15, 17, 20]
```

### Test Longer Contexts
```python
TOTAL_DOCS = 50  # Need more distractors
```

### Add Models
```python
MODELS = {
    "gemma-2b": "google/gemma-2-2b-it",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    # ...
}
```

## Why This Matters

- RAG systems often stuff many documents into context
- If critical info lands in the middle, models may ignore it
- Small models may be more affected than large ones
- Mitigations: reranking, chunking, recursive summarization

## References

- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) (Liu et al., 2023)
