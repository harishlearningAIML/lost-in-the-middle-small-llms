"""Tests for the model_runner module."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from src.model_runner import ModelRunner, DryRunModelRunner


class TestDryRunModelRunner:
    """Test the DryRunModelRunner (mock implementation)."""

    def test_initialization(self):
        """Should initialize without errors."""
        runner = DryRunModelRunner("/path/to/model")
        assert runner.model_path == "/path/to/model"

    def test_load(self, capsys):
        """Should handle load gracefully."""
        runner = DryRunModelRunner("/path/to/model")
        runner.load()

        captured = capsys.readouterr()
        assert "[DRY RUN]" in captured.out or "Would load" in captured.out

    def test_generate(self):
        """Should return mock response."""
        runner = DryRunModelRunner("/path/to/model")
        runner.load()

        response, latency = runner.generate("What is 2+2?")

        assert isinstance(response, str)
        assert "[DRY RUN]" in response or "no actual" in response.lower()
        assert isinstance(latency, (int, float))
        assert latency == 0.0

    def test_generate_with_parameters(self):
        """Should accept generation parameters."""
        runner = DryRunModelRunner("/path/to/model")
        runner.load()

        response, latency = runner.generate(
            "Question?", max_new_tokens=100, temperature=0.7
        )

        assert isinstance(response, str)
        assert isinstance(latency, (int, float))

    def test_unload(self):
        """Should unload gracefully."""
        runner = DryRunModelRunner("/path/to/model")
        runner.load()
        runner.unload()  # Should not raise


class TestModelRunner:
    """Test the ModelRunner class."""

    def test_initialization(self):
        """Should initialize with model path and device."""
        runner = ModelRunner("/path/to/model", device="cpu")

        assert runner.model_path == "/path/to/model"
        assert runner.device == "cpu"
        assert runner.model is None
        assert runner.tokenizer is None

    def test_device_auto_detection(self):
        """Should handle 'auto' device."""
        runner = ModelRunner("/path/to/model", device="auto")
        assert runner.device == "auto"

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_load_cpu(
        self, mock_tokenizer_from_pretrained, mock_model_from_pretrained, capsys
    ):
        """Should load model on CPU."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model_from_pretrained.return_value = mock_model

        runner = ModelRunner("/path/to/model", device="cpu")
        runner.load()

        # Verify model was loaded
        assert runner.model is not None
        assert runner.tokenizer is not None

        # Verify pad token was set
        mock_tokenizer_from_pretrained.assert_called_once()
        mock_model_from_pretrained.assert_called_once()

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_generate_not_loaded(self, mock_tok, mock_model):
        """Should raise error if generate called before load."""
        runner = ModelRunner("/path/to/model")

        with pytest.raises(RuntimeError):
            runner.generate("Test prompt")

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_generate_basic(
        self, mock_tokenizer_from_pretrained, mock_model_from_pretrained
    ):
        """Should generate response from prompt."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"

        # Mock tokenizer return
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = Mock(side_effect=lambda x: torch.tensor([[1, 2, 3]]))
        mock_inputs.to = Mock(return_value=mock_inputs)
        mock_tokenizer.return_value = mock_inputs

        # Mock tokenizer.apply_chat_template
        mock_tokenizer.apply_chat_template = Mock(return_value="formatted prompt")

        # Mock decode
        mock_tokenizer.decode = Mock(return_value="Test response")

        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_model.device = "cpu"

        # Mock generate output
        mock_output = torch.tensor([[1, 2, 3, 4, 5, 6]])  # More tokens than input
        mock_model.generate = Mock(return_value=mock_output)

        mock_model_from_pretrained.return_value = mock_model

        runner = ModelRunner("/path/to/model", device="cpu")
        runner.load()

        response, latency = runner.generate("Test prompt", max_new_tokens=50)

        assert isinstance(response, str)
        assert isinstance(latency, float)
        assert latency >= 0

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_unload(self, mock_tok, mock_model_class):
        """Should properly unload model."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"

        mock_model = MagicMock()
        mock_model.device = "cpu"

        mock_tok.return_value = mock_tokenizer
        mock_model_class.return_value = mock_model

        runner = ModelRunner("/path/to/model", device="cpu")
        runner.load()

        # Verify model is loaded
        assert runner.model is not None

        # Unload
        runner.unload()

        # Verify model is unloaded
        assert runner.model is None
        assert runner.tokenizer is None


class TestModelRunnerParameters:
    """Test model runner with different generation parameters."""

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_generate_with_temperature_zero(self, mock_tok, mock_model_class):
        """Should set do_sample=False when temperature=0."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.apply_chat_template = Mock(return_value="prompt")

        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = Mock(return_value=torch.tensor([[1, 2, 3]]))
        mock_inputs.to = Mock(return_value=mock_inputs)
        mock_tokenizer.return_value = mock_inputs

        mock_tokenizer.decode = Mock(return_value="response")

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))

        mock_tok.return_value = mock_tokenizer
        mock_model_class.return_value = mock_model

        runner = ModelRunner("/path/to/model", device="cpu")
        runner.load()

        runner.generate("prompt", temperature=0.0)

        # Verify do_sample was set to False
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["do_sample"] is False

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_generate_with_temperature_nonzero(self, mock_tok, mock_model_class):
        """Should set do_sample=True when temperature>0."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.apply_chat_template = Mock(return_value="prompt")

        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = Mock(return_value=torch.tensor([[1, 2, 3]]))
        mock_inputs.to = Mock(return_value=mock_inputs)
        mock_tokenizer.return_value = mock_inputs

        mock_tokenizer.decode = Mock(return_value="response")

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))

        mock_tok.return_value = mock_tokenizer
        mock_model_class.return_value = mock_model

        runner = ModelRunner("/path/to/model", device="cpu")
        runner.load()

        runner.generate("prompt", temperature=0.7)

        # Verify do_sample was set to True
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["do_sample"] is True
        assert call_kwargs["temperature"] == 0.7
