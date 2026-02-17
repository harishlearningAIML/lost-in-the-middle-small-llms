# Code Review: Lost in the Middle Experiment

**Date:** January 25, 2026  
**Reviewer:** Auto (AI Code Assistant)

## Executive Summary

Overall, the codebase is well-structured and functional. The experiment design is sound, and the code demonstrates good separation of concerns. However, there are several areas for improvement including bug fixes, error handling, code quality, and maintainability.

---

## 🔴 Critical Issues

### 1. **Config Import Inconsistency** (`run_experiment.py:22`)
**Issue:** The main script hardcodes import from `config_llama`:
```python
from config_llama import MODELS, POSITIONS, TOTAL_DOCS, TRIALS_PER_POSITION, MAX_NEW_TOKENS, TEMPERATURE
```

**Problem:** 
- This makes it impossible to use `config.py` or `config_gemma.py` without modifying the source code
- The README suggests using different configs, but the code doesn't support this

**Recommendation:**
- Add a `--config` CLI argument to select which config to use
- Or use environment variable to select config
- Or make config selection automatic based on `--model` argument

### 2. **Token Truncation Bug** (`model_runner.py:104`)
**Issue:** Hardcoded `max_length=8192` but contexts can be ~10K tokens:
```python
inputs = self.tokenizer(
    formatted,
    return_tensors="pt",
    truncation=True,
    max_length=8192  # ⚠️ Context can be ~10K tokens!
).to(self.model.device)
```

**Problem:** This silently truncates long contexts, potentially cutting off the gold document or question.

**Recommendation:**
- Remove `max_length` parameter (let model handle it)
- Or increase to model's actual context window (128K for Llama, 8K for Gemma)
- Add warning when truncation occurs

### 3. **Answer Extraction Bug** (`evaluator.py:27`)
**Issue:** Removing "the" prefix can break answers:
```python
prefixes = [
    "the answer is",
    "answer:",
    "the",  # ⚠️ This removes "the" from answers like "The Beatles"
    ...
] 
```

**Problem:** Answers like "The Beatles", "The Hague", "The Matrix" will become "Beatles", "Hague", "Matrix".

**Recommendation:**
- Remove "the" from the prefix list
- Or make it more specific: only remove "the" if followed by "answer is" or similar

---









## 🟡 Potential Bugs

### 4. **No Position Validation** (`context_builder.py:56`)
**Issue:** No check that `gold_position <= total_docs`:
```python
for i in range(1, total_docs + 1):
    if i == gold_position:  # ⚠️ What if gold_position > total_docs?
```

**Recommendation:**
```python
if gold_position > total_docs:
    raise ValueError(f"gold_position ({gold_position}) exceeds total_docs ({total_docs})")
```

### 5. **Insufficient Distractor Handling** (`context_builder.py:45`)
**Issue:** If `hard_distractors` + `generic_distractors` < `num_distractors_needed`, code will fail:
```python
remaining_needed = num_distractors_needed - len(all_distractors)
if remaining_needed > 0:
    all_distractors.extend(shuffled_generic[:remaining_needed])  # ⚠️ May not have enough
```

**Problem:** If there aren't enough distractors, `distractor_idx` will go out of bounds.

**Recommendation:**
- Add validation/error handling
- Or allow repeating distractors if needed

### 6. **Device Map Type Mismatch** (`model_runner.py:36`)
**Issue:** `device_map` can be string or "auto", but code assigns string to `device_map`:
```python
if torch.backends.mps.is_available():
    device_map = "mps"  # ⚠️ Should be device object, not string
```

**Problem:** `device_map="mps"` may not work correctly with `from_pretrained()`. Should use `device="mps"` or `device_map="auto"`.

**Recommendation:**
```python
if torch.backends.mps.is_available():
    device_map = "mps"  # or use device="mps" parameter instead
```

---

## 🟢 Code Quality Issues

### 7. **Unused Import** (`run_experiment.py:13`)
```python
import os  # ⚠️ Never used
```

### 8. **Magic Numbers**
- `max_length=8192` in `model_runner.py:104`
- `max_length=8192` should be configurable or removed
- Hash seed calculation `hash(...) % (2**32)` could use a constant

### 9. **Inconsistent Error Handling**
- `model_runner.py:94` catches all exceptions silently:
```python
except Exception:  # ⚠️ Too broad
    formatted = prompt
```

**Recommendation:** Catch specific exceptions or log the error.

### 10. **Missing Type Hints**
Several functions lack complete type hints:
- `load_data()` in `run_experiment.py`
- Return types in `visualize.py` functions

### 11. **Code Duplication**
- Config files (`config.py`, `config_gemma.py`, `config_llama.py`) have similar structure
- Consider a base config with model-specific overrides

---

## 📝 Documentation & Comments

### 12. **Incomplete Docstrings**
- `extract_answer()` doesn't document the normalization logic
- `build_context()` could better explain the hard distractor strategy
- Some functions lack examples

### 13. **README vs Code Mismatch**
- README mentions `config.py` but code uses `config_llama.py` by default
- README shows `--dry-run` but doesn't mention it's for testing

---

## 🚀 Best Practices

### 14. **Resource Management**
**Good:** Model unloading is implemented (`model_runner.py:128`)

**Improvement:** Consider using context manager:
```python
class ModelRunner:
    def __enter__(self):
        self.load()
        return self
    
    def __exit__(self, *args):
        self.unload()
```

### 15. **Logging**
**Issue:** Uses `print()` statements instead of proper logging

**Recommendation:** Use Python's `logging` module for better control and levels

### 16. **Configuration Management**
**Issue:** Hardcoded model paths in config files

**Recommendation:**
- Use environment variables for model paths
- Or config file with `.gitignore` for local paths
- Document required environment setup

### 17. **Testing**
**Issue:** No unit tests visible

**Recommendation:**
- Add tests for `evaluator.py` (already has test cases in `__main__`)
- Add tests for `context_builder.py`
- Add integration tests for full pipeline

### 18. **Data Validation**
**Issue:** No validation of loaded JSON data structure

**Recommendation:**
- Validate `qa_pairs.json` has required fields (`question`, `answer`, `gold_doc`, etc.)
- Validate `distractors.json` is a list of strings

---

## 🔧 Performance & Optimization

### 19. **Model Loading**
**Good:** Model is loaded once per experiment run

**Potential:** Could add option to keep model loaded across multiple experiments

### 20. **Memory Management**
**Good:** Model unloading implemented

**Improvement:** Consider `gc.collect()` after unloading for better memory cleanup

### 21. **Progress Tracking**
**Good:** Uses `tqdm` for progress bars

**Improvement:** Could add ETA and rate information

---

## 📊 Specific File Reviews

### `run_experiment.py`
- ✅ Good: Clear structure, good separation of concerns
- ⚠️ Issue: Hardcoded config import (Critical #1)
- ⚠️ Issue: Unused import (Quality #7)
- 💡 Suggestion: Add `--config` argument

### `model_runner.py`
- ✅ Good: Clean API, proper resource management
- 🔴 Critical: Token truncation bug (#2)
- ⚠️ Issue: Device map handling (#6)
- ⚠️ Issue: Broad exception handling (#9)

### `context_builder.py`
- ✅ Good: Clear logic, reproducible with seeds
- ⚠️ Issue: No position validation (#4)
- ⚠️ Issue: No distractor count validation (#5)

### `evaluator.py`
- ✅ Good: Handles multiple answer formats
- 🔴 Critical: "the" prefix removal bug (#3)
- 💡 Suggestion: Add more test cases

### `visualize.py`
- ✅ Good: Multiple visualization types
- ⚠️ Issue: Missing type hints
- 💡 Suggestion: Add error handling for missing data

### `config*.py`
- ⚠️ Issue: Code duplication (#11)
- ⚠️ Issue: Hardcoded paths (#16)

---

## ✅ What's Working Well

1. **Clear project structure** - Good separation of concerns
2. **Reproducibility** - Uses seeds for random operations
3. **Experiment design** - Well thought out with hard distractors
4. **Documentation** - README is comprehensive
5. **Visualization** - Good variety of charts
6. **Resource management** - Model unloading implemented
7. **CLI interface** - Good argument parsing

---

## 🎯 Priority Recommendations

### High Priority (Fix Soon)
1. Fix config import inconsistency (#1)
2. Fix token truncation bug (#2)
3. Fix "the" prefix removal bug (#3)
4. Add position validation (#4)

### Medium Priority (Next Sprint)
5. Improve error handling (#9)
6. Add data validation (#18)
7. Replace print with logging (#15)
8. Fix device map handling (#6)

### Low Priority (Nice to Have)
9. Add unit tests (#17)
10. Refactor config files (#11)
11. Add type hints (#10)
12. Use context managers (#14)

---

## 📋 Quick Fix Checklist

- [ ] Change `run_experiment.py` to support config selection
- [ ] Remove or increase `max_length` in `model_runner.py`
- [ ] Fix "the" prefix removal in `evaluator.py`
- [ ] Add position validation in `context_builder.py`
- [ ] Remove unused `os` import
- [ ] Add error handling for insufficient distractors
- [ ] Fix device_map assignment
- [ ] Add logging instead of print statements
- [ ] Add data validation on JSON load
- [ ] Document config selection in README

---

## 🔍 Additional Observations

1. **Version Control**: Many files are modified but not committed (per git status)
2. **File Naming**: `linekding.txt` appears to be a typo
3. **Data Structure**: Consider using dataclasses or Pydantic for QA pairs
4. **Results Format**: JSON structure is good, but could add schema validation

---

## Conclusion

The codebase is functional and well-organized for an experimental project. The main issues are:
- **Critical bugs** that could affect results (truncation, answer extraction)
- **Configuration flexibility** that limits usability
- **Error handling** that could mask problems

Addressing the high-priority items will significantly improve reliability and maintainability.

**Overall Assessment: 7/10** - Good foundation, needs bug fixes and improvements for production use.
