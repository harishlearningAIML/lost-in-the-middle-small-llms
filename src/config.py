"""Configuration for Lost in the Middle experiment - V3."""

import os

# Model paths - use env vars for portability
# Set LOST_IN_MIDDLE_GEMMA_2B_PATH, LOST_IN_MIDDLE_GEMMA_4B_PATH, LOST_IN_MIDDLE_LLAMA_3B_PATH
_DEFAULT_PATHS = {
    "gemma-2b": "/Volumes/T9/models/gemma-2-2b-it-v2",
    "gemma-4b": "/Volumes/T9/models/gemma-3-4b-it-v1",
    "llama-3b": "/Volumes/T9/models/llama-3.2-3b-instruct-v1",
}

MODELS = {
    name: os.environ.get(
        f"LOST_IN_MIDDLE_{name.upper().replace('-', '_')}_PATH", path
    )
    for name, path in _DEFAULT_PATHS.items()
}

# Model-specific config: Llama degrades at 100 docs (MPS), Gemma handles 100
MODEL_CONFIG = {
    "gemma-2b": {
        "positions": [1, 10, 25, 50, 75, 90, 100],
        "total_docs": 100,
    },
    "gemma-4b": {
        "positions": [1, 10, 25, 50, 75, 90, 100],
        "total_docs": 100,
    },
    "llama-3b": {
        "positions": [1, 10, 25, 35, 50, 60, 70],
        "total_docs": 70,
    },
}

# Shared settings
TRIALS_PER_POSITION = 100
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.0
