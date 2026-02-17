#!/usr/bin/env python3
"""
Complete RAG Pipeline with Attention-Aware Optimizations

This module provides a complete pipeline that:
1. Takes retrieved documents from your vector DB
2. Optionally re-ranks them
3. Reorders based on target model's attention pattern
4. Builds an optimized prompt
5. Returns the final prompt ready for inference

Example:
    >>> from solutions.pipeline import RAGPipeline
    >>>
    >>> pipeline = RAGPipeline(model="gemma-2b")
    >>> prompt = pipeline.process(
    ...     query="What is the capital of France?",
    ...     documents=retrieved_docs
    ... )
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

from .reorder import reorder_documents
from .prompts import build_rag_prompt, RAGPromptBuilder, PromptConfig


@dataclass
class PipelineConfig:
    """Configuration for the RAG pipeline."""

    # Target model for optimization
    model: str = "gemma-2b"

    # Maximum documents to include in context
    max_documents: int = 10

    # Key in document dict containing text
    text_key: str = "text"

    # Key in document dict containing relevance score
    score_key: str = "score"

    # Enable document reordering based on model attention
    enable_reordering: bool = True

    # Enable attention refresher (repeat query at end)
    enable_reminder: bool = True

    # Require citation before answer
    require_citation: bool = False

    # Custom re-ranker function (optional)
    reranker: Optional[Callable[[List[Dict], str], List[Dict]]] = None


class RAGPipeline:
    """
    Complete RAG pipeline with attention-aware optimizations.

    This pipeline implements all the strategies discovered from
    the "Lost in the Middle" experiments on small LLMs.
    """

    def __init__(
        self,
        model: str = "gemma-2b",
        max_documents: int = 10,
        enable_reordering: bool = True,
        enable_reminder: bool = True,
        require_citation: bool = False,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            model: Target model ("gemma-2b", "gemma-4b", "llama-3b")
            max_documents: Maximum docs to include in context
            enable_reordering: Apply attention-aware reordering
            enable_reminder: Repeat query at end of context
            require_citation: Force model to cite before answering
        """
        self.config = PipelineConfig(
            model=model,
            max_documents=max_documents,
            enable_reordering=enable_reordering,
            enable_reminder=enable_reminder,
            require_citation=require_citation,
        )

    def set_reranker(self, reranker: Callable[[List[Dict], str], List[Dict]]) -> "RAGPipeline":
        """
        Set a custom re-ranker function.

        The re-ranker should take (documents, query) and return
        documents with updated scores.

        Args:
            reranker: Function to re-rank documents

        Returns:
            Self for chaining
        """
        self.config.reranker = reranker
        return self

    def process(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        text_key: Optional[str] = None,
        score_key: Optional[str] = None,
    ) -> str:
        """
        Process a query and documents through the pipeline.

        Steps:
        1. Filter to max_documents
        2. Re-rank if reranker is set
        3. Reorder based on model attention pattern
        4. Build optimized prompt

        Args:
            query: User's question
            documents: Retrieved documents from vector DB
            text_key: Override config text_key
            score_key: Override config score_key

        Returns:
            Complete prompt string ready for model inference
        """
        text_key = text_key or self.config.text_key
        score_key = score_key or self.config.score_key

        # Step 1: Limit documents
        docs = documents[: self.config.max_documents]

        # Step 2: Re-rank if configured
        if self.config.reranker is not None:
            docs = self.config.reranker(docs, query)

        # Step 3: Reorder based on model attention
        if self.config.enable_reordering:
            docs = reorder_documents(docs, model=self.config.model, score_key=score_key)

        # Step 4: Build prompt
        prompt_config = PromptConfig(
            include_reminder=self.config.enable_reminder,
            require_citation=self.config.require_citation,
            max_context_docs=self.config.max_documents,
        )

        return build_rag_prompt(query, docs, text_key, prompt_config)

    def process_with_details(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Process query and return detailed pipeline output.

        Returns dict with:
        - prompt: The final prompt string
        - original_order: Document IDs in original order
        - reordered: Document IDs after reordering
        - config: Pipeline configuration used

        Useful for debugging and understanding the pipeline.
        """
        score_key = self.config.score_key
        text_key = self.config.text_key

        # Track original order
        docs = documents[: self.config.max_documents]
        original_order = [d.get("id", i) for i, d in enumerate(docs)]

        # Re-rank
        if self.config.reranker is not None:
            docs = self.config.reranker(docs, query)

        after_rerank = [d.get("id", i) for i, d in enumerate(docs)]

        # Reorder
        if self.config.enable_reordering:
            docs = reorder_documents(docs, model=self.config.model, score_key=score_key)

        after_reorder = [d.get("id", i) for i, d in enumerate(docs)]

        # Build prompt
        prompt = self.process(query, documents)

        return {
            "prompt": prompt,
            "original_order": original_order,
            "after_rerank": after_rerank,
            "after_reorder": after_reorder,
            "config": {
                "model": self.config.model,
                "max_documents": self.config.max_documents,
                "enable_reordering": self.config.enable_reordering,
                "enable_reminder": self.config.enable_reminder,
                "require_citation": self.config.require_citation,
            },
        }


def create_pipeline(model: str, **kwargs) -> RAGPipeline:
    """
    Factory function to create a pre-configured pipeline.

    Args:
        model: Target model name
        **kwargs: Additional config options

    Returns:
        Configured RAGPipeline
    """
    # Model-specific defaults
    defaults = {
        "gemma-2b": {
            "enable_reordering": True,
            "enable_reminder": True,  # Important for recency bias
            "require_citation": False,
            "max_documents": 10,
        },
        "gemma-4b": {
            "enable_reordering": True,  # Uses best-last (recency bias like gemma-2b)
            "enable_reminder": True,
            "require_citation": False,
            "max_documents": 15,
        },
        "llama-3b": {
            "enable_reordering": False,  # Stable performance, standard order fine
            "enable_reminder": False,
            "require_citation": False,
            "max_documents": 10,
        },
    }

    config = defaults.get(model.lower(), defaults["gemma-2b"])
    config.update(kwargs)

    return RAGPipeline(model=model, **config)


# Example re-ranker using simple keyword matching (for demo)
def simple_reranker(documents: List[Dict], query: str) -> List[Dict]:
    """
    Simple keyword-based re-ranker for demonstration.

    In production, use a proper re-ranker like:
    - BGE-M3
    - Cohere Rerank
    - sentence-transformers cross-encoder
    """
    query_words = set(query.lower().split())

    for doc in documents:
        text = doc.get("text", "").lower()
        overlap = sum(1 for word in query_words if word in text)
        doc["rerank_score"] = overlap + doc.get("score", 0)

    return sorted(documents, key=lambda x: x.get("rerank_score", 0), reverse=True)


if __name__ == "__main__":
    # Demo the pipeline
    sample_docs = [
        {"id": 1, "text": "The capital of Valdoria is Zenith City.", "score": 0.95},
        {"id": 2, "text": "Valdoria is located in the northern hemisphere.", "score": 0.72},
        {"id": 3, "text": "The population of Valdoria is 5 million.", "score": 0.68},
        {"id": 4, "text": "Valdoria's currency is the Valdorian Crown.", "score": 0.65},
        {"id": 5, "text": "Zenith City was founded in 1823.", "score": 0.60},
    ]

    query = "What is the capital of Valdoria?"

    print("=" * 60)
    print("RAG PIPELINE DEMO")
    print("=" * 60)

    # Create pipeline for Gemma-2B
    pipeline = create_pipeline("gemma-2b")

    # Process with details
    result = pipeline.process_with_details(query, sample_docs)

    print(f"\nOriginal order: {result['original_order']}")
    print(f"After reorder:  {result['after_reorder']}")
    print(f"\nConfig: {result['config']}")

    print("\n" + "-" * 60)
    print("FINAL PROMPT:")
    print("-" * 60)
    print(result["prompt"])
