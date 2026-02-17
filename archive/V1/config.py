"""
Configuration for Lost in the Middle experiment
"""

# Model configurations - adjust paths as needed
MODELS = {
    "gemma-2b": "/Volumes/T9/models/gemma-2-2b-it-v2",
    "gemma-4b": "/Volumes/T9/models/gemma-3-4b-it-v1", 
    "llama-3b": "/Volumes/T9/models/llama-3.2-3b-instruct-v1",
}

# Experiment settings
POSITIONS = [1, 5, 10, 15, 20]  # Where to place gold document
TOTAL_DOCS = 20                  # Total documents in context
TRIALS_PER_POSITION = 20         # How many QA pairs to test per position

# Generation settings
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.0  # Greedy decoding for reproducibility

# Paths
DATA_DIR = "data"
RESULTS_DIR = "results"
