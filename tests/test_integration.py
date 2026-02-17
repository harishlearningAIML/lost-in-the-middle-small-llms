"""Integration tests for the full pipeline."""

import pytest
import json
from pathlib import Path

from src.context_builder import build_context, build_prompt
from src.evaluator import check_answer
from src.model_runner import DryRunModelRunner


class TestFullPipeline:
    """Test complete pipeline from context building to evaluation."""

    def test_build_and_evaluate(self, sample_qa_pair, sample_distractors):
        """Test building context, prompt, and evaluating an answer."""
        # Build context
        context = build_context(
            sample_qa_pair, sample_distractors, gold_position=1, total_docs=5, seed=42
        )

        # Build prompt
        prompt = build_prompt(context, sample_qa_pair["question"])

        # Simulate model response
        response = f"The answer is {sample_qa_pair['answer']}"

        # Evaluate
        is_correct, extracted = check_answer(response, sample_qa_pair["answer"])

        assert is_correct is True
        assert extracted  # Should have extracted something

    def test_multiple_positions(self, sample_qa_pair, sample_distractors):
        """Test that different positions produce different contexts but same answer."""
        positions = [1, 3, 5]
        contexts = []

        for pos in positions:
            context = build_context(
                sample_qa_pair,
                sample_distractors,
                gold_position=pos,
                total_docs=5,
                seed=42,
            )
            contexts.append(context)

            # Verify gold doc is at correct position
            lines = context.split("\n\n")
            assert sample_qa_pair["gold_doc"] in lines[pos - 1]

        # Contexts should be different
        assert contexts[0] != contexts[1]
        assert contexts[1] != contexts[2]

    def test_model_runner_dry_run(self, sample_qa_pair, sample_distractors):
        """Test dry run model with context."""
        context = build_context(
            sample_qa_pair, sample_distractors, gold_position=1, total_docs=5, seed=42
        )

        prompt = build_prompt(context, sample_qa_pair["question"])

        runner = DryRunModelRunner("/mock/model")
        runner.load()

        response, latency = runner.generate(prompt, max_new_tokens=50)

        assert isinstance(response, str)
        assert latency == 0.0

        runner.unload()


class TestDataLoadingIntegration:
    """Test integration with actual data files."""

    def test_load_fixture_data(self, qa_pairs_fixture, distractors_fixture):
        """Should load fixture data correctly."""
        assert isinstance(qa_pairs_fixture, list)
        assert len(qa_pairs_fixture) > 0

        # Check first QA pair structure
        qa = qa_pairs_fixture[0]
        assert "id" in qa
        assert "question" in qa
        assert "answer" in qa
        assert "gold_doc" in qa

        # Check distractors
        assert isinstance(distractors_fixture, list)
        assert len(distractors_fixture) > 0
        assert all(isinstance(d, str) for d in distractors_fixture)

    def test_build_context_with_fixture_data(
        self, qa_pairs_fixture, distractors_fixture
    ):
        """Should successfully build context with fixture data."""
        qa = qa_pairs_fixture[0]

        context = build_context(
            qa, distractors_fixture, gold_position=1, total_docs=5, seed=42
        )

        assert context
        assert qa["gold_doc"] in context
        assert "Document 1:" in context
        assert "Document 5:" in context


class TestErrorHandling:
    """Test error handling in pipeline."""

    def test_invalid_gold_position_greater_than_total(
        self, sample_qa_pair, sample_distractors
    ):
        """Should handle position > total_docs gracefully."""
        # This might raise or handle gracefully depending on implementation
        # At minimum, the test documents the behavior
        try:
            context = build_context(
                sample_qa_pair,
                sample_distractors,
                gold_position=100,
                total_docs=5,
                seed=42,
            )
            # If it doesn't raise, verify gold doc is still there
            assert sample_qa_pair["gold_doc"] in context
        except (ValueError, IndexError):
            # If it raises, that's also acceptable
            pass

    def test_insufficient_distractors(self, sample_qa_pair):
        """Should handle insufficient distractors."""
        insufficient_distractors = ["Only one distractor"]

        try:
            context = build_context(
                sample_qa_pair,
                insufficient_distractors,
                gold_position=1,
                total_docs=10,
                seed=42,
            )
            # If successful, verify structure
            assert context
            assert sample_qa_pair["gold_doc"] in context
        except (ValueError, IndexError):
            # If it raises, that's acceptable
            pass

    def test_check_answer_with_empty_response(self, sample_qa_pair):
        """Should handle empty response gracefully."""
        is_correct, extracted = check_answer("", sample_qa_pair["answer"])

        assert is_correct is False
        assert isinstance(extracted, str)

    def test_check_answer_with_none_values(self, sample_qa_pair):
        """Should handle None or missing values gracefully."""
        # Empty strings instead of None
        is_correct, extracted = check_answer("", "")

        assert is_correct is False
        assert extracted == ""


class TestReproducibility:
    """Test reproducibility of experiments."""

    def test_seed_reproducibility(self, sample_qa_pair, sample_distractors):
        """Should produce identical results with same seed."""
        seed = 12345

        results = []
        for _ in range(3):
            context = build_context(
                sample_qa_pair,
                sample_distractors,
                gold_position=5,
                total_docs=10,
                seed=seed,
            )

            prompt = build_prompt(context, sample_qa_pair["question"])
            results.append(prompt)

        # All results should be identical
        assert results[0] == results[1]
        assert results[1] == results[2]

    def test_different_seed_different_results(self, sample_qa_pair, sample_distractors):
        """Should produce different results with different seeds."""
        context1 = build_context(
            sample_qa_pair, sample_distractors, gold_position=5, total_docs=10, seed=1
        )

        context2 = build_context(
            sample_qa_pair, sample_distractors, gold_position=5, total_docs=10, seed=2
        )

        # Contexts should be different due to different distractor shuffling
        assert context1 != context2

        # But gold doc should be at same position
        lines1 = context1.split("\n\n")
        lines2 = context2.split("\n\n")

        assert sample_qa_pair["gold_doc"] in lines1[4]  # Position 5 (0-indexed)
        assert sample_qa_pair["gold_doc"] in lines2[4]


class TestEdgeCases:
    """Test edge cases across pipeline."""

    def test_very_long_question(self, sample_qa_pair, sample_distractors):
        """Should handle very long questions."""
        long_question = "What " * 100 + "is the answer?"

        context = build_context(
            sample_qa_pair, sample_distractors, gold_position=1, total_docs=3, seed=42
        )

        prompt = build_prompt(context, long_question)

        assert long_question in prompt

    def test_unicode_in_data(self, sample_distractors):
        """Should handle unicode characters."""
        qa_unicode = {
            "id": "test_unicode",
            "question": "Quelle est la capitale? (What is the capital?)",
            "answer": "Zürich",
            "gold_doc": "Zürich is a beautiful city in Switzerland.",
            "hard_distractors": [
                "Genève is also in Switzerland.",
                "Bern is the capital.",
            ],
        }

        context = build_context(
            qa_unicode, sample_distractors, gold_position=1, total_docs=5, seed=42
        )

        assert "Zürich" in context
        assert "Switzerland" in context

    def test_special_characters_in_answer(self, sample_distractors):
        """Should handle special characters in answers."""
        qa_special = {
            "id": "test_special",
            "question": "What is the formula?",
            "answer": "E=mc²",
            "gold_doc": "Einstein's formula is E=mc².",
            "hard_distractors": [],
        }

        is_correct, extracted = check_answer("E=mc²", "E=mc²")

        # Should match or at least not crash
        assert isinstance(is_correct, bool)
        assert isinstance(extracted, str)
