"""Configuration for Lost in the Middle experiment - Gemma models (2B & 4B)."""

# Model paths
MODELS = {
    "gemma-2b": "/Volumes/T9/models/gemma-2-2b-it-v2",
    "gemma-4b": "/Volumes/T9/models/gemma-3-4b-it-v1",
}

# Experiment settings - Gemma handles 100 docs well
POSITIONS = [1, 10, 25, 50, 75, 90, 100]  # 7 positions including deep middle
TOTAL_DOCS = 100  # 100 documents per context (~10K tokens)
TRIALS_PER_POSITION = 30  # 30 trials for statistical significance

# Generation settings
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.0  # Deterministic
