"""Configuration for Lost in the Middle experiment - Llama 3B specific."""

# Model paths
MODELS = {
    "llama-3b": "/Volumes/T9/models/llama-3.2-3b-instruct-v1",
}

# Experiment settings - Llama works best with fewer docs (70 max)
POSITIONS = [1, 10, 25, 35, 50, 60, 70]  # 7 positions scaled to 70 docs
TOTAL_DOCS = 70  # 70 documents (~7K tokens) - Llama's sweet spot
TRIALS_PER_POSITION = 30  # 30 trials for statistical significance

# Generation settings
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.0  # Deterministic
