"""Pytest configuration and shared fixtures."""

import json
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def qa_pairs_fixture(fixtures_dir):
    """Load test QA pairs."""
    with open(fixtures_dir / "qa_pairs.json") as f:
        return json.load(f)


@pytest.fixture
def distractors_fixture(fixtures_dir):
    """Load test distractors."""
    with open(fixtures_dir / "distractors.json") as f:
        return json.load(f)


@pytest.fixture
def sample_qa_pair():
    """Single QA pair for testing."""
    return {
        "id": "test_1",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "gold_doc": "Paris is the capital of France.",
        "hard_distractors": [
            "London is the capital of England.",
            "Berlin is the capital of Germany.",
        ],
    }


@pytest.fixture
def sample_distractors():
    """Sample distractor list."""
    return [
        "The Earth is round.",
        "Water boils at 100°C.",
        "The Sun is a star.",
        "Dogs are mammals.",
        "Trees produce oxygen.",
        "The sky is blue.",
        "Grass is green.",
        "Fish live in water.",
        "Birds can fly.",
        "Insects have six legs.",
    ] * 5  # Repeat to have enough distractors


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"
    tokenizer.encode = Mock(return_value=[1, 2, 3])
    tokenizer.decode = Mock(return_value="Test response")
    tokenizer.apply_chat_template = Mock(return_value="formatted prompt")
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = MagicMock()
    model.device = "cpu"

    # Mock generate method
    mock_output = MagicMock()
    mock_output.__getitem__ = Mock(side_effect=lambda x: [1, 2, 3, 4, 5])
    mock_output.__len__ = Mock(return_value=1)

    model.generate = Mock(return_value=mock_output)
    model.parameters = Mock(return_value=iter([]))

    return model


@pytest.fixture
def mock_model_runner(mock_model, mock_tokenizer):
    """Mock ModelRunner for integration tests."""
    from src.model_runner import ModelRunner

    runner = ModelRunner("mock-model", device="cpu")
    runner.model = mock_model
    runner.tokenizer = mock_tokenizer

    return runner
