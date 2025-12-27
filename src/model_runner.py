"""Model runner for HuggingFace inference."""

import time
from typing import Tuple
import torch


class ModelRunner:
    """Handles model loading and inference."""

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the model runner.

        Args:
            model_path: Path to HuggingFace model
            device: Device to use ("auto", "cuda", "mps", "cpu")
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load the model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model from {self.model_path}...")
        start = time.time()

        # Determine device and dtype - use bfloat16 like V1
        if self.device == "auto":
            if torch.cuda.is_available():
                device_map = "auto"
            elif torch.backends.mps.is_available():
                device_map = "mps"
            else:
                device_map = "cpu"
        else:
            device_map = self.device

        # Always use bfloat16 (worked in V1)
        torch_dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        elapsed = time.time() - start
        print(f"Model loaded on {device_map} in {elapsed:.1f}s")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ) -> Tuple[str, float]:
        """
        Generate a response for the given prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for greedy)

        Returns:
            Tuple of (response_text, latency_ms)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.time()

        # Format as chat if model expects it
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                formatted = prompt
        else:
            formatted = prompt

        # Tokenize
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        ).to(self.model.device)

        # Generate (match V1 approach exactly)
        do_sample = temperature > 0

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        latency_ms = (time.time() - start_time) * 1000

        # Decode only new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip(), latency_ms

    def unload(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Unloaded model")


class DryRunModelRunner:
    """Mock model runner for testing without GPU."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        
    def load(self):
        print(f"[DRY RUN] Would load model from {self.model_path}")
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ) -> Tuple[str, float]:
        return "[DRY RUN - no actual inference]", 0.0
    
    def unload(self):
        pass


if __name__ == "__main__":
    # Test with dry run
    runner = DryRunModelRunner("/path/to/model")
    runner.load()
    response, latency = runner.generate("What is 2+2?")
    print(f"Response: {response}")
    print(f"Latency: {latency:.2f}ms")
