"""Configuration for Lost in the Middle experiment - V3 (100 docs, more trials)."""

# Model paths - UPDATE THESE to your local paths
MODELS = {
    "gemma-2b": "/Volumes/T9/models/gemma-2-2b-it-v2",
    "gemma-4b": "/Volumes/T9/models/gemma-3-4b-it-v1",
    "llama-3b": "/Volumes/T9/models/llama-3.2-3b-instruct-v1",
}

# Experiment settings - V3: 100 docs, more positions
POSITIONS = [1, 10, 25, 50, 75, 90, 100]  # 7 positions including deep middle
TOTAL_DOCS = 100                           # 100 documents per context (~10K tokens)
TRIALS_PER_POSITION = 30                   # 30 trials for statistical significance

# Generation settings
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.0  # Deterministic
