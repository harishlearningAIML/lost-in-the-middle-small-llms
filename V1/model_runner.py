"""
Model Runner - Load and run inference on small open source models
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import time


class ModelRunner:
    """Wrapper for running inference on HuggingFace models"""
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """Load model and tokenizer"""
        print(f"Loading {self.model_name}...")
        start = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        elapsed = time.time() - start
        print(f"Loaded in {elapsed:.1f}s")
        return self
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        do_sample: bool = False
    ) -> str:
        """Generate response for a prompt"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Format as chat if model expects it
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                formatted = prompt
        else:
            formatted = prompt
        
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=8192  # Leave room for generation
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def unload(self):
        """Free GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        print(f"Unloaded {self.model_name}")


# Quick test
if __name__ == "__main__":
    # Test with a small model
    runner = ModelRunner("google/gemma-2-2b-it")
    runner.load()
    
    prompt = "What is 2 + 2? Answer with just the number."
    response = runner.generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    runner.unload()
