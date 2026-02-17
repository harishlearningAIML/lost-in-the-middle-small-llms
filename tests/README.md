# Test Suite Documentation

This directory contains the comprehensive test suite for the Lost in the Middle project.

## Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── test_evaluator.py        # Tests for answer evaluation
├── test_context_builder.py  # Tests for context building
├── test_model_runner.py     # Tests for model inference
├── test_integration.py      # End-to-end integration tests
└── fixtures/
    ├── qa_pairs.json        # Sample QA pairs for testing
    └── distractors.json     # Sample distractor documents
```

## Running Tests

### Install test dependencies
```bash
pip install -r requirements-dev.txt
```

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_evaluator.py
```

### Run specific test class
```bash
pytest tests/test_evaluator.py::TestNormalize
```

### Run specific test function
```bash
pytest tests/test_evaluator.py::TestNormalize::test_lowercase
```

### Run with verbose output
```bash
pytest -v
```

### Run with coverage report
```bash
pytest --cov=src --cov-report=html
```

### Run only unit tests (exclude integration)
```bash
pytest -m "not integration"
```

### Run tests in parallel
```bash
pytest -n auto
```

## Test Coverage

### test_evaluator.py
- **Normalize function**: lowercase, punctuation removal, whitespace normalization
- **Extract answer function**: prefix removal, trailing punctuation, multi-word extraction
- **Check answer function**: exact matches, case insensitivity, number matching, edge cases
- **Integration patterns**: realistic model responses (Gemma, Llama styles)

### test_context_builder.py
- **Build context**: document placement, gold position validation, distractor inclusion
- **Deterministic behavior**: seed reproducibility, different seeds produce different contexts
- **Document structure**: correct numbering, no gaps, proper formatting
- **Edge cases**: empty hard distractors, missing keys, single document

### test_model_runner.py
- **Dry run runner**: initialization, loading, generation, unloading
- **Model runner**: loading, generation with parameters, device handling
- **Inference parameters**: temperature handling, do_sample flag
- **Error handling**: not loaded errors, proper cleanup

### test_integration.py
- **Full pipeline**: context building → prompt → evaluation
- **Multiple positions**: different gold positions produce valid contexts
- **Data loading**: fixture data integration
- **Reproducibility**: seed-based reproducibility across runs
- **Error handling**: missing data, invalid positions
- **Edge cases**: unicode, very long inputs, special characters

## Key Fixtures (conftest.py)

- `qa_pairs_fixture`: Loads test QA pairs from JSON
- `distractors_fixture`: Loads test distractors from JSON
- `sample_qa_pair`: Single QA pair for unit tests
- `sample_distractors`: List of sample distractors
- `mock_tokenizer`: Mocked tokenizer for model tests
- `mock_model`: Mocked model for generation tests
- `mock_model_runner`: Full mocked ModelRunner instance

## Best Practices

1. **Use fixtures**: Fixtures provide consistent test data and mocks
2. **Isolate units**: Tests should verify single responsibilities
3. **Test edge cases**: Empty inputs, unicode, large data, special chars
4. **Use meaningful names**: Test names should describe what is being tested
5. **Group related tests**: Use test classes to organize related tests
6. **Mock external dependencies**: Use unittest.mock for transformers/torch

## Adding New Tests

1. Create test function starting with `test_`
2. Use existing fixtures or create new ones in `conftest.py`
3. Follow naming convention: `test_module_description`
4. Add docstring explaining what is being tested
5. Use assertions with meaningful messages when possible

Example:
```python
def test_my_feature(sample_qa_pair, sample_distractors):
    """Should do something specific."""
    result = build_context(sample_qa_pair, sample_distractors, ...)
    assert result is not None
```

## Continuous Integration

These tests are designed to work in CI/CD pipelines:

```bash
# Install deps
pip install -r requirements-dev.txt

# Run tests with coverage
pytest --cov=src --cov-report=xml tests/

# Run linting
flake8 src/ --max-line-length=100
mypy src/ --ignore-missing-imports

# Format check
black --check src/ tests/
```
