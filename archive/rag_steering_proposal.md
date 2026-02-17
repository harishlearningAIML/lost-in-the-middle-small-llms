# RAG Position-Aware Steering Proposal

**Based on:** "Lost in the Middle" findings + Valanyx Enforcement Engine steering logic

---

## Key Findings to Address

From the validation, we know:
- ✅ **Small models (2-4B) show recency bias**: Better performance when relevant docs are at the END
- ✅ **Early positions (1-10) are WORST**: 83-87% accuracy (Gemma models)
- ✅ **Late positions (75-100) are BEST**: 90-97% accuracy (Gemma models)
- ✅ **Position matters more than primacy**: Contradicts "put important stuff first" wisdom

---

## Current Valanyx Steering Architecture

### 1. **Activation Steering** (Lines 403-424, 596-616)
```python
def _get_orthogonal_vector(self, steering_vec, hidden_state):
    """Dynamically aligns dimensions and calculates Orthogonal Vector"""
    # 1. Dynamic dimension alignment
    # 2. Orthogonal projection to avoid disrupting existing semantics
    # 3. Returns steering vector that guides without overpowering
```

**Mechanism:**
- Injects semantic guidance into hidden states at specific layers
- Uses orthogonal projection to preserve original information
- Dynamic intensity based on alignment between hidden state and steering vector

### 2. **Safe Zones** (Lines 225-232)
```python
self.safe_zones = {
    "DEFINITION": (0.40, 0.55),      # Early-middle layers
    "REGULATORY": (0.55, 0.70),      # Middle layers
    "PROCEDURAL": (0.55, 0.75),      # Middle-late layers
    "EXCEPTION": (0.85, 0.95),       # Very late layers
    "DEFAULT": (0.55, 0.70)          # Middle layers
}
```

**Insight:** Different rule types target different layer ranges
- Early layers (40-55%): Foundational concepts
- Middle layers (55-75%): Reasoning and logic
- Late layers (85-95%): Final decision-making

### 3. **Multi-Tiered Intensity** (Lines 878-884)
```python
if score >= self.retrieval_threshold:
    active_intensity = profile['intensity']        # Full force
else:
    active_intensity = profile['intensity'] * 0.3  # Ghost mode
```

**Insight:** Steering strength adapts to confidence level

### 4. **Knowledge Base Retrieval** (Lines 844-868)
- Domain-based filtering
- Semantic similarity matching
- Returns top-k most relevant rules

---

## Proposed: Position-Aware Steering for RAG

### Core Idea
**Compensate for small model position bias by applying position-adaptive steering**

Instead of treating all document positions equally, apply:
1. **Stronger steering** for early positions (where models struggle)
2. **Lighter steering** for late positions (where models naturally excel)
3. **Position-aware layer targeting** (early docs → early layers, late docs → late layers)

---

## Strategy 1: Position-Graduated Intensity

### Implementation
```python
def _calculate_position_intensity(self, base_intensity, position, total_docs):
    """
    Calculate steering intensity based on document position.

    Early positions (1-10): Boost by 50% to compensate for weakness
    Middle positions (25-50): Use base intensity
    Late positions (75-100): Reduce by 30% (let natural recency work)
    """
    position_pct = position / total_docs

    if position_pct < 0.15:  # Early (0-15%)
        return base_intensity * 1.5  # BOOST to compensate
    elif position_pct < 0.50:  # Early-Middle (15-50%)
        return base_intensity * 1.2
    elif position_pct < 0.75:  # Middle-Late (50-75%)
        return base_intensity * 1.0  # Baseline
    else:  # Late (75-100%)
        return base_intensity * 0.7  # Let recency bias work naturally
```

**Why this works:**
- Early positions get stronger steering to overcome model's natural weakness
- Late positions rely more on model's innate recency bias
- Middle positions get balanced treatment

### Validation Data Support
| Position Range | Current Accuracy | Proposed Intensity | Expected Improvement |
|----------------|------------------|-------------------|---------------------|
| 1-10 | 83-87% (worst) | 1.5x base | +5-7% (boost weak positions) |
| 25-50 | 90-93% | 1.0-1.2x base | +2-3% (maintain) |
| 75-100 | 90-97% (best) | 0.7x base | 0% (already optimal) |

---

## Strategy 2: Position-Aware Layer Targeting

### Implementation
```python
def _get_position_aware_safe_zone(self, position, total_docs, rule_type):
    """
    Map document position to optimal steering layer range.

    Early positions: Steer at EARLY layers (inject context early)
    Late positions: Steer at LATE layers (reinforce final decision)
    """
    position_pct = position / total_docs

    if position_pct < 0.20:  # Early documents
        # Inject at early layers (40-55%) - foundational level
        return (0.40, 0.55)
    elif position_pct < 0.60:  # Middle documents
        # Standard middle layers (55-70%) - reasoning level
        return (0.55, 0.70)
    else:  # Late documents
        # Target late layers (70-85%) - decision level
        return (0.70, 0.85)
```

**Rationale:**
- **Early docs → Early layers**: Inject context before model "forgets"
- **Late docs → Late layers**: Reinforce what model naturally attends to
- Aligns steering with model's natural attention patterns

### Example for 26-layer model:
```
Position 1-20:   Steer at layers 10-14 (early semantic processing)
Position 50:     Steer at layers 14-18 (middle reasoning)
Position 75-100: Steer at layers 18-22 (late decision-making)
```

---

## Strategy 3: Retrieval + Smart Reordering

### Current Approach (Sub-optimal)
```
1. Retrieve top-k most relevant documents
2. Place them in order of similarity score (highest first)
3. Result: Best doc at position 1 (WORST performance)
```

### Proposed: Recency-Optimized Ordering
```python
def reorder_for_recency_bias(self, retrieved_docs, strategy="reverse"):
    """
    Reorder retrieved documents to place most important ones LAST.

    Strategies:
    - 'reverse': Simply reverse order (best doc last)
    - 'sandwich': Important at start AND end, distractors in middle
    - 'graduated': Exponentially increase importance toward end
    """
    if strategy == "reverse":
        # Simple: Best document at the END
        return list(reversed(retrieved_docs))

    elif strategy == "sandwich":
        # U-curve mitigation: Good docs at both ends
        # Format: [2nd-best, distractors..., best]
        if len(retrieved_docs) >= 3:
            best = retrieved_docs[0]
            second = retrieved_docs[1]
            rest = retrieved_docs[2:]
            return [second] + rest + [best]
        return list(reversed(retrieved_docs))

    elif strategy == "graduated":
        # Exponential increase in importance
        # Format: [weak, weak, medium, strong, strongest]
        return sorted(retrieved_docs, key=lambda x: x['score'])

    return retrieved_docs
```

**Validation Evidence:**
- Gemma-4B: Position 100 = 96.7%, Position 1 = 86.7% (+10% by moving to end!)
- Gemma-2B: Position 100 = 93.3%, Position 1 = 86.7% (+6.7%)
- Simple reordering can yield 7-10% accuracy boost for FREE

---

## Strategy 4: Hybrid Approach (Reordering + Adaptive Steering)

### Combined Architecture
```python
def rag_generate_with_position_awareness(self, query, retrieved_docs):
    """
    Full position-aware RAG pipeline.

    1. Retrieve relevant documents
    2. Reorder to place best docs at END (recency optimization)
    3. Apply position-graduated steering to each doc
    4. Generate with position-adaptive intensity
    """
    # Step 1: Retrieve (existing logic)
    docs = self.retrieve_documents(query, top_k=10)

    # Step 2: Reorder for recency bias
    docs = self.reorder_for_recency_bias(docs, strategy="reverse")

    # Step 3: Build context with position tracking
    context_parts = []
    for position, doc in enumerate(docs, start=1):
        context_parts.append(f"Document {position}: {doc['text']}")

    full_context = "\n\n".join(context_parts)

    # Step 4: Apply position-aware steering for the BEST document
    best_doc = docs[-1]  # Now at the end!
    best_position = len(docs)

    # Calculate adaptive intensity
    base_intensity = self.topology['steering_profiles']['DEFAULT']['intensity']
    position_intensity = self._calculate_position_intensity(
        base_intensity, best_position, len(docs)
    )

    # Get position-aware layer range
    safe_zone = self._get_position_aware_safe_zone(
        best_position, len(docs), best_doc.get('type', 'DEFAULT')
    )

    # Register steering hooks with position-aware parameters
    steering_vec = self.embedder.encode(best_doc['text'], convert_to_tensor=True)

    for logical_layer in [0, 1, 2]:  # Top 3 layers in safe zone
        physical_layer = self._translate_to_safe_zone(logical_layer, safe_zone)
        self._register_hook(physical_layer, position_intensity, steering_vec)

    # Step 5: Generate with steering active
    response = self.model.generate(query, context=full_context)

    return response
```

**Key Innovations:**
1. ✅ Reordering places best doc at position 100 (natural +7-10% boost)
2. ✅ Reduced steering intensity (0.7x) since position is already optimal
3. ✅ Late-layer targeting (70-85%) aligns with model's attention
4. ✅ Combines natural bias + strategic steering for maximum effect

---

## Strategy 5: Dynamic Position Detection

### Challenge
In real RAG, you don't always know the total document count upfront.

### Solution: Relative Position Scoring
```python
def _get_relative_position_score(self, doc_index, total_docs):
    """
    Calculate relative position score (0.0 = start, 1.0 = end).

    Use this to dynamically adjust steering without knowing total upfront.
    """
    return doc_index / total_docs if total_docs > 0 else 0.5

def adaptive_steering_from_position(self, doc, doc_index, context_size):
    """
    Apply steering that adapts to document's relative position.
    """
    rel_pos = self._get_relative_position_score(doc_index, context_size)

    # Inverse intensity: early = high, late = low
    intensity_multiplier = 1.0 - (rel_pos * 0.5)  # Range: 1.0 → 0.5

    # Layer shift: early = early layers, late = late layers
    base_zone_start, base_zone_end = self.safe_zones['DEFAULT']
    shifted_start = base_zone_start + (rel_pos * 0.20)  # Shift up by up to 20%
    shifted_end = base_zone_end + (rel_pos * 0.15)      # Shift up by up to 15%

    return intensity_multiplier, (shifted_start, shifted_end)
```

---

## Expected Performance Gains

### Conservative Estimate (Strategy 1: Reordering Only)
| Model | Current Avg | With Reordering | Gain |
|-------|------------|----------------|------|
| Gemma-2B | 89.5% | **94.0%** | +4.5% |
| Gemma-4B | 91.9% | **96.5%** | +4.6% |
| Llama-3B | 94.8% | **96.5%** | +1.7% |

**Cost:** Zero - just reorder array before building prompt

### Optimistic Estimate (Strategy 4: Hybrid)
| Model | Current Avg | With Hybrid | Gain |
|-------|------------|-------------|------|
| Gemma-2B | 89.5% | **95-96%** | +5.5-6.5% |
| Gemma-4B | 91.9% | **97-98%** | +5.1-6.1% |
| Llama-3B | 94.8% | **97%** | +2.2% |

**Cost:** +50-100ms latency (position-aware steering computation)

---

## Implementation Checklist

### Phase 1: Low-Hanging Fruit (Reordering)
- [ ] Add `reorder_for_recency_bias()` function
- [ ] Modify retrieval pipeline to reverse document order
- [ ] A/B test: Original order vs Reversed order
- [ ] Measure accuracy improvement

### Phase 2: Position-Aware Steering
- [ ] Add `_calculate_position_intensity()` function
- [ ] Add `_get_position_aware_safe_zone()` function
- [ ] Integrate position parameters into `_register_hook()`
- [ ] Log position-steering mappings for analysis

### Phase 3: Validation
- [ ] Run "Lost in the Middle" experiment with position-aware steering
- [ ] Compare: No steering, Standard steering, Position-aware steering
- [ ] Measure accuracy by position with each approach

### Phase 4: Optimization
- [ ] Profile latency impact of position calculations
- [ ] Optimize layer selection algorithm
- [ ] Cache position-to-layer mappings
- [ ] Benchmark memory usage

---

## Code Integration Points

### In `enforcement_engine.py`

**1. Add position-aware intensity calculation:**
```python
# Around line 878, replace:
if score >= self.retrieval_threshold:
    active_intensity = profile['intensity']
else:
    active_intensity = profile['intensity'] * 0.3

# With:
if score >= self.retrieval_threshold:
    base_intensity = profile['intensity']
    # Apply position-aware multiplier
    active_intensity = self._calculate_position_intensity(
        base_intensity, document_position, total_documents
    )
else:
    active_intensity = profile['intensity'] * 0.3
```

**2. Add position-aware layer targeting:**
```python
# Around line 889, replace:
for logic_l in profile['logical_layers']:
    phys_l = self.translate_logical_to_physical(logic_l, rtype)
    self._register_hook(phys_l, active_intensity, s_vec)

# With:
position_zone = self._get_position_aware_safe_zone(
    document_position, total_documents, rtype
)
for logic_l in profile['logical_layers']:
    phys_l = self.translate_logical_to_physical_with_zone(
        logic_l, position_zone
    )
    self._register_hook(phys_l, active_intensity, s_vec)
```

**3. Add document reordering in retrieval:**
```python
# Around line 856, after retrieval:
rule = retrieve_rule_util(query, self.embedder, filtered_embeddings,
                         filtered_kb, threshold=self.fail_safe_threshold)

# Add reordering for position optimization:
if self.enable_position_aware_rag:
    # Place highest-scoring document at END for recency bias
    # (Reverses the natural "best first" order)
    document_position = len(filtered_kb)  # Assume placed last
else:
    document_position = 1  # Traditional: best doc first
```

---

## Validation Methodology

### Experiment Design
Rerun "Lost in the Middle" experiment with 3 conditions:

1. **Baseline (Original)**: Best doc at position 1
2. **Reordered**: Best doc at position 100
3. **Position-Aware Steering**: Best doc at position 100 + adaptive steering

### Metrics to Track
- Accuracy by position (1, 10, 25, 50, 75, 90, 100)
- Latency impact (steering computation time)
- Memory usage (additional hooks/tensors)
- Error rates on hard questions (q1, q4, q21)

### Success Criteria
- ✅ Accuracy improvement of ≥3% on average
- ✅ Latency increase of ≤100ms
- ✅ No degradation on already-good positions (75-100)
- ✅ Significant improvement on weak positions (1-10)

---

## Risks & Mitigations

### Risk 1: Over-steering at Late Positions
**Problem:** Late positions already perform well (90-97%), additional steering might hurt

**Mitigation:**
- Use reduced intensity (0.7x) for positions >75%
- A/B test to find optimal intensity curve
- Add "no steering" mode for positions >90%

### Risk 2: Latency Impact
**Problem:** Position calculations add compute overhead

**Mitigation:**
- Pre-compute position-to-layer mappings at init time
- Cache steering vectors per position
- Use lightweight intensity calculations (simple math, no models)

### Risk 3: Model-Specific Behavior
**Problem:** Llama showed different pattern than Gemma

**Mitigation:**
- Make position curves model-configurable
- Add topology setting: `position_bias_type: "recency" | "u_curve" | "flat"`
- Auto-detect bias pattern during warmup

---

## Theoretical Foundation

### Why Position-Aware Steering Works

**1. Attention Mechanics:**
- Small models have limited attention capacity
- Early tokens compete for attention with later tokens
- Steering at early layers helps "anchor" early-positioned information
- Steering at late layers reinforces what model already attends to

**2. Information Flow:**
```
Early Position + Early Layer Steering:
  → Information injected BEFORE it gets lost
  → Compensates for attention decay

Late Position + Minimal Steering:
  → Model naturally attends here (recency bias)
  → Steering would be redundant/harmful
```

**3. Orthogonal Projection:**
- Current steering uses orthogonal vectors to avoid overwriting
- Position-aware approach AMPLIFIES this for weak positions
- REDUCES for strong positions (let model do its thing)

### Alignment with "Lost in the Middle" Findings

| Finding | How Steering Helps |
|---------|-------------------|
| Early positions worst (83%) | Boost intensity 1.5x + early layer steering |
| Middle positions OK (90%) | Baseline intensity + middle layer steering |
| Late positions best (97%) | Reduce intensity 0.7x + late layer steering |
| Recency bias pattern | Leverage natural bias, don't fight it |

---

## Alternative: Attention Bias Injection

### More Advanced Approach
Instead of steering hidden states, directly modify attention weights:

```python
def inject_position_bias_to_attention(self, attention_weights, doc_positions):
    """
    Directly modify attention weights to boost end-positioned documents.

    More invasive but potentially more effective than hidden state steering.
    """
    bias_curve = torch.linspace(0.5, 1.5, len(doc_positions))  # 0.5x early → 1.5x late

    for i, pos in enumerate(doc_positions):
        attention_weights[:, pos] *= bias_curve[i]

    # Renormalize
    attention_weights = torch.softmax(attention_weights, dim=-1)

    return attention_weights
```

**Pros:**
- Direct control over what model attends to
- Surgical precision (target specific document positions)

**Cons:**
- More invasive (modifies attention mechanism)
- Model-specific (attention architecture varies)
- Harder to debug

---

## Recommended Implementation Order

1. **Start Simple**: Implement Strategy 1 (Reordering only)
   - Easiest to implement
   - Zero latency cost
   - Expected +4-7% improvement

2. **Add Adaptive Intensity**: Implement Strategy 1 + position-graduated intensity
   - Moderate complexity
   - ~20ms latency cost
   - Expected additional +1-2% improvement

3. **Add Layer Targeting**: Full Strategy 4 (Hybrid)
   - Most complex
   - ~50-100ms latency cost
   - Expected additional +1-2% improvement

4. **Experiment with Attention**: Strategy 5 (if needed)
   - Research-grade complexity
   - Unknown latency impact
   - Potential for breakthrough gains

---

## Summary: Key Takeaways

### What We Learned from "Lost in the Middle"
1. ✅ Small models have **recency bias** (not U-curve)
2. ✅ Position 1-10: **WORST** (83-87% accuracy)
3. ✅ Position 75-100: **BEST** (90-97% accuracy)
4. ✅ Effect size: **+7-10%** improvement by moving doc to end

### How Valanyx Steering Can Help
1. ✅ **Reorder documents** to place important ones at END (free +7-10%)
2. ✅ **Position-graduated intensity** to compensate for weak positions
3. ✅ **Position-aware layer targeting** to align with attention patterns
4. ✅ **Hybrid approach** for maximum performance (expected +5-7% total)

### Next Steps
1. **Quick Win**: Add document reordering (1 hour implementation)
2. **Medium Win**: Add position-aware intensity (4 hours implementation)
3. **Full Win**: Implement hybrid approach (1-2 days implementation)
4. **Validation**: Rerun "Lost in the Middle" experiment to measure gains

---

**Bottom Line:** By leveraging the validated findings that small models have recency bias, we can strategically place documents and apply position-aware steering to achieve **5-10% accuracy improvements** with minimal latency cost.
