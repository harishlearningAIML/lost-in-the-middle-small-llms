#!/usr/bin/env python3
"""
Prompt Templates for Small LLMs with Attention Issues

These templates implement the "Attention Refresher" pattern:
- Repeat the query at the end of the context
- Force citation before answering
- Structure generation to improve retrieval from long contexts
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    include_reminder: bool = True
    require_citation: bool = True
    max_context_docs: int = 10
    doc_separator: str = "\n---\n"


def format_documents(
    documents: List[Dict[str, Any]],
    text_key: str = "text",
    include_index: bool = True
) -> str:
    """
    Format documents into a string for inclusion in prompts.

    Args:
        documents: List of document dicts
        text_key: Key containing document text
        include_index: Whether to number documents [1], [2], etc.

    Returns:
        Formatted string of all documents
    """
    lines = []
    for i, doc in enumerate(documents, 1):
        text = doc.get(text_key, str(doc))
        if include_index:
            lines.append(f"[Document {i}]\n{text}")
        else:
            lines.append(text)
    return "\n\n".join(lines)


def build_rag_prompt(
    query: str,
    documents: List[Dict[str, Any]],
    text_key: str = "text",
    config: Optional[PromptConfig] = None
) -> str:
    """
    Build a RAG prompt with attention refresher pattern.

    This template:
    1. States the task clearly
    2. Presents context documents
    3. Repeats the query (attention refresher)
    4. Optionally requires citation before answering

    Args:
        query: User's question
        documents: Retrieved context documents
        text_key: Key in document dict containing text
        config: Prompt configuration options

    Returns:
        Complete prompt string

    Example:
        >>> docs = [{"text": "Paris is the capital of France."}]
        >>> prompt = build_rag_prompt("What is the capital of France?", docs)
    """
    if config is None:
        config = PromptConfig()

    # Limit documents if needed
    docs = documents[:config.max_context_docs]

    # Format context
    context = format_documents(docs, text_key)

    # Build prompt parts
    parts = []

    # System instruction
    parts.append(
        "Answer the user's question based ONLY on the following context documents. "
        "If the answer is not in the documents, say \"I don't know.\""
    )

    # Context section
    parts.append(f"\n\n=== CONTEXT ===\n{context}\n=== END CONTEXT ===")

    # Attention refresher - repeat the query
    if config.include_reminder:
        parts.append(f"\n\nReminder: The user asked: \"{query}\"")

    # Citation requirement
    if config.require_citation:
        parts.append(
            "\n\nFirst, quote the exact sentence from the documents that answers this question. "
            "Then, provide your final answer."
        )
        parts.append(f"\n\nQuestion: {query}")
        parts.append("\nRelevant quote: ")
    else:
        parts.append(f"\n\nQuestion: {query}")
        parts.append("\nAnswer: ")

    return "".join(parts)


def build_rag_prompt_simple(
    query: str,
    documents: List[Dict[str, Any]],
    text_key: str = "text"
) -> str:
    """
    Build a simple RAG prompt without fancy formatting.

    Good for baseline comparisons or when you want minimal overhead.

    Args:
        query: User's question
        documents: Retrieved context documents
        text_key: Key in document dict containing text

    Returns:
        Simple prompt string
    """
    context = format_documents(documents, text_key, include_index=False)

    return f"""Context:
{context}

Question: {query}
Answer:"""


def build_rag_prompt_gemma(
    query: str,
    documents: List[Dict[str, Any]],
    text_key: str = "text"
) -> str:
    """
    Build a RAG prompt optimized for Gemma models.

    Uses Gemma's chat template format and places the query
    at the very end (exploiting recency bias).

    Args:
        query: User's question
        documents: Retrieved context documents
        text_key: Key in document dict containing text

    Returns:
        Gemma-optimized prompt string
    """
    context = format_documents(documents, text_key)

    # Gemma instruction format with query repeated at end
    return f"""<start_of_turn>user
I need you to answer a question using ONLY the information in the documents below.

Documents:
{context}

IMPORTANT: Answer this question: {query}

If the answer is not in the documents, respond with "Information not found."
<end_of_turn>
<start_of_turn>model
Based on the documents provided, """


def build_rag_prompt_llama(
    query: str,
    documents: List[Dict[str, Any]],
    text_key: str = "text"
) -> str:
    """
    Build a RAG prompt optimized for Llama models.

    Uses Llama's instruction format. Since Llama-3B showed
    stable performance, this uses standard ordering.

    Args:
        query: User's question
        documents: Retrieved context documents
        text_key: Key in document dict containing text

    Returns:
        Llama-optimized prompt string
    """
    context = format_documents(documents, text_key)

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that answers questions based only on the provided context.
If the answer is not in the context, say "I don't know."<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def build_cot_prompt(
    query: str,
    documents: List[Dict[str, Any]],
    text_key: str = "text"
) -> str:
    """
    Build a Chain-of-Thought prompt for complex questions.

    Forces the model to:
    1. Identify relevant documents
    2. Extract key information
    3. Reason step by step
    4. Provide final answer

    This helps with attention by making retrieval explicit.

    Args:
        query: User's question
        documents: Retrieved context documents
        text_key: Key in document dict containing text

    Returns:
        CoT prompt string
    """
    context = format_documents(documents, text_key)

    return f"""Answer the following question using the documents below.

Documents:
{context}

Question: {query}

Let's solve this step by step:
1. Which document(s) contain information relevant to this question?
2. What specific facts from those documents help answer the question?
3. Based on these facts, what is the answer?

Step 1 - Relevant documents:"""


class RAGPromptBuilder:
    """
    Fluent builder for RAG prompts with customization options.

    Example:
        >>> builder = RAGPromptBuilder()
        >>> prompt = (builder
        ...     .with_query("What is the capital?")
        ...     .with_documents(docs)
        ...     .with_reminder()
        ...     .require_citation()
        ...     .for_model("gemma-2b")
        ...     .build())
    """

    def __init__(self):
        self._query: str = ""
        self._documents: List[Dict[str, Any]] = []
        self._text_key: str = "text"
        self._include_reminder: bool = False
        self._require_citation: bool = False
        self._model: str = "generic"
        self._system_prompt: str = ""

    def with_query(self, query: str) -> "RAGPromptBuilder":
        """Set the user query."""
        self._query = query
        return self

    def with_documents(self, documents: List[Dict[str, Any]], text_key: str = "text") -> "RAGPromptBuilder":
        """Set the context documents."""
        self._documents = documents
        self._text_key = text_key
        return self

    def with_reminder(self, enabled: bool = True) -> "RAGPromptBuilder":
        """Enable/disable query reminder at end."""
        self._include_reminder = enabled
        return self

    def require_citation(self, enabled: bool = True) -> "RAGPromptBuilder":
        """Enable/disable citation requirement."""
        self._require_citation = enabled
        return self

    def for_model(self, model: str) -> "RAGPromptBuilder":
        """Set target model for format optimization."""
        self._model = model.lower()
        return self

    def with_system_prompt(self, prompt: str) -> "RAGPromptBuilder":
        """Set custom system prompt."""
        self._system_prompt = prompt
        return self

    def build(self) -> str:
        """Build the final prompt string."""
        # Use model-specific templates if available
        if self._model in ("gemma-2b", "gemma-4b", "gemma"):
            return build_rag_prompt_gemma(self._query, self._documents, self._text_key)
        elif self._model in ("llama-3b", "llama"):
            return build_rag_prompt_llama(self._query, self._documents, self._text_key)

        # Generic prompt with options
        config = PromptConfig(
            include_reminder=self._include_reminder,
            require_citation=self._require_citation
        )
        return build_rag_prompt(self._query, self._documents, self._text_key, config)


if __name__ == "__main__":
    # Demo
    sample_docs = [
        {"text": "The capital of Valdoria is Zenith City, established in 1823."},
        {"text": "Valdoria is a country in the northern hemisphere with 5 million people."},
        {"text": "The Valdorian Crown is the official currency of Valdoria."},
    ]
    query = "What is the capital of Valdoria?"

    print("=" * 60)
    print("SIMPLE PROMPT")
    print("=" * 60)
    print(build_rag_prompt_simple(query, sample_docs))

    print("\n" + "=" * 60)
    print("RAG PROMPT WITH ATTENTION REFRESHER")
    print("=" * 60)
    print(build_rag_prompt(query, sample_docs))

    print("\n" + "=" * 60)
    print("GEMMA-OPTIMIZED PROMPT")
    print("=" * 60)
    print(build_rag_prompt_gemma(query, sample_docs))

    print("\n" + "=" * 60)
    print("CHAIN-OF-THOUGHT PROMPT")
    print("=" * 60)
    print(build_cot_prompt(query, sample_docs))
