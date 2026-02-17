"""
Solutions for "Lost in the Middle" attention issues in small LLMs.

This package provides strategies to work around positional biases
discovered in small language models (Gemma-2B, Gemma-4B, Llama-3B).

Modules:
    reorder: Document reordering strategies based on model attention patterns
    prompts: Prompt templates with attention refresher patterns
    pipeline: Complete RAG pipeline with integrated solutions

Usage:
    from solutions import reorder_documents, build_rag_prompt

    # Reorder documents for your target model
    docs = reorder_documents(retrieved_docs, model="gemma-2b")

    # Build an optimized prompt
    prompt = build_rag_prompt(query, docs)
"""

from .reorder import (
    reorder_documents,
    reorder_for_gemma_2b,
    reorder_for_gemma_4b,
    reorder_for_llama_3b,
    best_last,
    sides_first,
    best_first,
)

from .prompts import (
    build_rag_prompt,
    build_rag_prompt_simple,
    build_rag_prompt_gemma,
    build_rag_prompt_llama,
    build_cot_prompt,
    RAGPromptBuilder,
    PromptConfig,
)

from .pipeline import (
    RAGPipeline,
    PipelineConfig,
    create_pipeline,
)

__all__ = [
    # Reordering
    "reorder_documents",
    "reorder_for_gemma_2b",
    "reorder_for_gemma_4b",
    "reorder_for_llama_3b",
    "best_last",
    "sides_first",
    "best_first",
    # Prompts
    "build_rag_prompt",
    "build_rag_prompt_simple",
    "build_rag_prompt_gemma",
    "build_rag_prompt_llama",
    "build_cot_prompt",
    "RAGPromptBuilder",
    "PromptConfig",
    # Pipeline
    "RAGPipeline",
    "PipelineConfig",
    "create_pipeline",
]

__version__ = "1.0.0"
