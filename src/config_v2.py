"""Configuration for Lost in the Middle experiment - V2 (Harder)."""

# Model paths - UPDATE THESE to your local paths
MODELS = {
    "gemma-2b": "/Volumes/T9/models/gemma-2-2b-it-v2",
    "gemma-4b": "/Volumes/T9/models/gemma-3-4b-it-v1",
    "llama-3b": "/Volumes/T9/models/llama-3.2-3b-instruct-v1",
}

# Experiment settings - HARDER VERSION
POSITIONS = [1, 10, 25, 40, 50]  # Gold doc positions to test
TOTAL_DOCS = 50                   # 50 documents per context (was 20)
TRIALS_PER_POSITION = 20          # Questions per position

# Generation settings
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.0  # Deterministic
