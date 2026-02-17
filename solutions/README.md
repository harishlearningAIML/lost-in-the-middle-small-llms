# Solutions for "Lost in the Middle" in Small LLMs

This package provides practical solutions for working around positional attention biases discovered in small language models.

## Experimental Findings

| Model | Bias Pattern | Worst Position | Best Position | p-value |
|-------|-------------|----------------|---------------|---------|
| **Gemma-2B** | Recency Bias | Position 10 (82%) | Position 100 (97%) | 0.0225* |
| **Gemma-4B** | Weak Lost in Middle | Position 50 (89%) | Position 100 (97%) | 0.198 |
| **Llama-3B** | Flat/Stable | Position 50 (89%) | Position 1 (94%) | 1.0 |

\* Statistically significant (p < 0.05)

## Quick Start

```python
from solutions import reorder_documents, build_rag_prompt

# Your retrieved documents from vector DB
docs = [
    {"text": "The capital of France is Paris.", "score": 0.95},
    {"text": "France is in Europe.", "score": 0.82},
    {"text": "Paris has the Eiffel Tower.", "score": 0.78},
]

# Reorder for your target model
optimized_docs = reorder_documents(docs, model="gemma-2b")

# Build prompt with attention refresher
prompt = build_rag_prompt("What is the capital of France?", optimized_docs)
```

## Strategies

### 1. Document Reordering

Since you can't change the model's attention pattern, change where you place important documents.

#### Gemma-2B: "Best-Last" Strategy

```python
from solutions import reorder_for_gemma_2b, best_last

# Both are equivalent
docs = best_last(retrieved_docs)
docs = reorder_for_gemma_2b(retrieved_docs)

# Result: [worst, ..., middle, ..., best]
# The most relevant document is now LAST, where Gemma-2B pays most attention
```

#### Gemma-4B: "Sides-First" Strategy

```python
from solutions import reorder_for_gemma_4b, sides_first

docs = sides_first(retrieved_docs)

# Result: [best, 3rd, 5th, ..., 4th, 2nd]
# Best documents at edges, lower-confidence in the middle
```

#### Llama-3B: Standard Order

```python
from solutions import reorder_for_llama_3b, best_first

docs = best_first(retrieved_docs)

# Result: [best, 2nd, 3rd, ...]
# Standard descending order - Llama-3B handles all positions similarly
```

### 2. Attention Refresher Prompts

Small models often "forget" the instruction after reading many documents. Repeat the query at the end.

```python
from solutions import build_rag_prompt, PromptConfig

config = PromptConfig(
    include_reminder=True,   # Repeat query at end
    require_citation=True,   # Force citation before answer
    max_context_docs=10
)

prompt = build_rag_prompt(query, docs, config=config)
```

**Generated prompt structure:**
```
Answer the user's question based ONLY on the following context...

=== CONTEXT ===
[Document 1] ...
[Document 2] ...
=== END CONTEXT ===

Reminder: The user asked: "What is the capital of France?"  <-- ATTENTION REFRESHER

First, quote the exact sentence...  <-- CITATION REQUIREMENT

Question: What is the capital of France?
Relevant quote:
```

### 3. Model-Specific Prompts

```python
from solutions import build_rag_prompt_gemma, build_rag_prompt_llama

# Gemma format (with special tokens)
prompt = build_rag_prompt_gemma(query, docs)

# Llama format (with special tokens)
prompt = build_rag_prompt_llama(query, docs)
```

### 4. Chain-of-Thought for Complex Questions

```python
from solutions import build_cot_prompt

prompt = build_cot_prompt(query, docs)

# Forces step-by-step reasoning:
# 1. Which documents are relevant?
# 2. What facts help answer the question?
# 3. What is the final answer?
```

### 5. Complete Pipeline

```python
from solutions.pipeline import RAGPipeline, create_pipeline

# Quick setup with model-specific defaults
pipeline = create_pipeline("gemma-2b")

# Or customize
pipeline = RAGPipeline(
    model="gemma-2b",
    max_documents=10,
    enable_reordering=True,
    enable_reminder=True,
    require_citation=False
)

# Process query
prompt = pipeline.process(query, retrieved_docs)

# Get detailed output for debugging
result = pipeline.process_with_details(query, retrieved_docs)
print(f"Original order: {result['original_order']}")
print(f"After reorder: {result['after_reorder']}")
```

## Integration Example

### With LangChain

```python
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from solutions import reorder_documents, build_rag_prompt

# Retrieve documents
vectorstore = Chroma(...)
docs = vectorstore.similarity_search_with_score(query, k=20)

# Convert to our format
formatted_docs = [
    {"text": doc.page_content, "score": score}
    for doc, score in docs
]

# Reorder for model
optimized = reorder_documents(formatted_docs, model="gemma-2b")

# Build prompt
prompt = build_rag_prompt(query, optimized)

# Run inference
llm = HuggingFacePipeline(...)
response = llm(prompt)
```

### With Custom Re-ranker

```python
from solutions.pipeline import RAGPipeline

def cohere_reranker(docs, query):
    # Your Cohere/BGE-M3 reranking logic
    import cohere
    co = cohere.Client("your-api-key")

    results = co.rerank(
        query=query,
        documents=[d["text"] for d in docs],
        top_n=10
    )

    reranked = []
    for r in results:
        doc = docs[r.index].copy()
        doc["score"] = r.relevance_score
        reranked.append(doc)

    return reranked

pipeline = RAGPipeline(model="gemma-2b")
pipeline.set_reranker(cohere_reranker)

prompt = pipeline.process(query, docs)
```

## Recommended Configurations

### Gemma-2B (Edge Deployment)

```python
pipeline = RAGPipeline(
    model="gemma-2b",
    max_documents=10,        # Reduce context for faster inference
    enable_reordering=True,  # Critical - exploit recency bias
    enable_reminder=True,    # Critical - refresh attention
    require_citation=False   # Keep it simple
)
```

### Gemma-4B (Quality Focus)

```python
pipeline = RAGPipeline(
    model="gemma-4b",
    max_documents=15,
    enable_reordering=True,  # Sides-first for lost-in-middle
    enable_reminder=True,
    require_citation=True    # Improves accuracy on complex questions
)
```

### Llama-3B (Balanced)

```python
pipeline = RAGPipeline(
    model="llama-3b",
    max_documents=10,
    enable_reordering=False,  # Not needed - stable performance
    enable_reminder=False,
    require_citation=False
)
```

## Files

- `reorder.py` - Document reordering strategies
- `prompts.py` - Prompt templates and builders
- `pipeline.py` - Complete RAG pipeline
- `__init__.py` - Package exports

## Testing

```bash
# Run the demos
python -m solutions.reorder
python -m solutions.prompts
python -m solutions.pipeline
```

## References

- Original Paper: "Lost in the Middle: How Language Models Use Long Contexts" (Liu et al., 2023)
- This Experiment: Testing positional biases in small open-source LLMs (Gemma-2B, Gemma-4B, Llama-3B)
