"""Tests for the context_builder module."""

import pytest
from src.context_builder import build_context, build_prompt


class TestBuildContext:
    """Test the build_context function."""

    def test_basic_context_building(self, sample_qa_pair, sample_distractors):
        """Should build context with gold doc at specified position."""
        context = build_context(
            sample_qa_pair, sample_distractors, gold_position=1, total_docs=5, seed=42
        )

        assert isinstance(context, str)
        assert sample_qa_pair["gold_doc"] in context
        assert "Document 1:" in context
        assert "Document 5:" in context

    def test_gold_position_at_end(self, sample_qa_pair, sample_distractors):
        """Should correctly place gold doc at end position."""
        total_docs = 10
        context = build_context(
            sample_qa_pair,
            sample_distractors,
            gold_position=total_docs,
            total_docs=total_docs,
            seed=42,
        )

        # Extract last document
        lines = context.split("\n\n")
        last_doc = lines[-1]
        assert sample_qa_pair["gold_doc"] in last_doc
        assert f"Document {total_docs}:" in last_doc

    def test_gold_position_in_middle(self, sample_qa_pair, sample_distractors):
        """Should correctly place gold doc in middle position."""
        total_docs = 10
        mid_pos = 5
        context = build_context(
            sample_qa_pair,
            sample_distractors,
            gold_position=mid_pos,
            total_docs=total_docs,
            seed=42,
        )

        lines = context.split("\n\n")
        mid_doc = lines[mid_pos - 1]  # 0-indexed
        assert sample_qa_pair["gold_doc"] in mid_doc
        assert f"Document {mid_pos}:" in mid_doc

    def test_hard_distractors_included(self, sample_qa_pair, sample_distractors):
        """Should include hard distractors in context."""
        context = build_context(
            sample_qa_pair, sample_distractors, gold_position=5, total_docs=10, seed=42
        )

        # Check that at least one hard distractor is in context
        hard_distractors = sample_qa_pair.get("hard_distractors", [])
        assert any(distractor in context for distractor in hard_distractors)

    def test_deterministic_shuffling_with_seed(
        self, sample_qa_pair, sample_distractors
    ):
        """Should produce same context with same seed."""
        context1 = build_context(
            sample_qa_pair, sample_distractors, gold_position=5, total_docs=10, seed=42
        )

        context2 = build_context(
            sample_qa_pair, sample_distractors, gold_position=5, total_docs=10, seed=42
        )

        assert context1 == context2

    def test_different_seeds_produce_different_contexts(
        self, sample_qa_pair, sample_distractors
    ):
        """Should produce different contexts with different seeds."""
        context1 = build_context(
            sample_qa_pair, sample_distractors, gold_position=5, total_docs=10, seed=42
        )

        context2 = build_context(
            sample_qa_pair, sample_distractors, gold_position=5, total_docs=10, seed=123
        )

        # Contexts should differ (distractors shuffled differently)
        assert context1 != context2
        # But gold doc should still be at position 5
        assert sample_qa_pair["gold_doc"] in context1
        assert sample_qa_pair["gold_doc"] in context2

    def test_document_count(self, sample_qa_pair, sample_distractors):
        """Should have correct number of documents."""
        total_docs = 20
        context = build_context(
            sample_qa_pair,
            sample_distractors,
            gold_position=10,
            total_docs=total_docs,
            seed=42,
        )

        doc_count = context.count("Document ")
        assert doc_count == total_docs

    def test_no_missing_documents(self, sample_qa_pair, sample_distractors):
        """Should have documents numbered 1 to total_docs without gaps."""
        total_docs = 15
        context = build_context(
            sample_qa_pair,
            sample_distractors,
            gold_position=8,
            total_docs=total_docs,
            seed=42,
        )

        for i in range(1, total_docs + 1):
            assert f"Document {i}:" in context

    def test_single_document(self, sample_qa_pair, sample_distractors):
        """Should work with single document (gold_position=1, total_docs=1)."""
        context = build_context(
            sample_qa_pair, sample_distractors, gold_position=1, total_docs=1, seed=42
        )

        assert sample_qa_pair["gold_doc"] in context
        assert "Document 1:" in context
        assert "Document 2:" not in context

    def test_sufficient_distractors(self, sample_qa_pair, sample_distractors):
        """Should handle case with sufficient generic distractors."""
        context = build_context(
            sample_qa_pair, sample_distractors, gold_position=5, total_docs=10, seed=42
        )

        # Should complete without error
        assert context
        assert len(context) > 0


class TestBuildContextEdgeCases:
    """Edge case tests for build_context."""

    def test_more_hard_distractors_than_needed(
        self, sample_qa_pair, sample_distractors
    ):
        """Should handle more hard distractors than needed for total_docs."""
        qa_many_hard = sample_qa_pair.copy()
        qa_many_hard["hard_distractors"] = sample_distractors[:5]

        context = build_context(
            qa_many_hard, sample_distractors, gold_position=1, total_docs=3, seed=42
        )

        assert context
        assert "Document 1:" in context
        assert "Document 2:" in context
        assert "Document 3:" in context

    def test_empty_hard_distractors(self, sample_qa_pair, sample_distractors):
        """Should work with empty hard_distractors list."""
        qa_no_hard = sample_qa_pair.copy()
        qa_no_hard["hard_distractors"] = []

        context = build_context(
            qa_no_hard, sample_distractors, gold_position=5, total_docs=10, seed=42
        )

        assert context
        assert "Document 5:" in context

    def test_missing_hard_distractors_key(self, sample_qa_pair, sample_distractors):
        """Should work when hard_distractors key is missing."""
        qa_no_key = sample_qa_pair.copy()
        del qa_no_key["hard_distractors"]

        context = build_context(
            qa_no_key, sample_distractors, gold_position=5, total_docs=10, seed=42
        )

        assert context
        assert sample_qa_pair["gold_doc"] in context


class TestBuildPrompt:
    """Test the build_prompt function."""

    def test_prompt_structure(self, sample_qa_pair, sample_distractors):
        """Should build well-formed prompt with context and question."""
        context = build_context(
            sample_qa_pair, sample_distractors, gold_position=1, total_docs=5, seed=42
        )

        prompt = build_prompt(context, sample_qa_pair["question"])

        assert isinstance(prompt, str)
        assert "Document 1:" in prompt
        assert sample_qa_pair["question"] in prompt
        assert "Answer:" in prompt

    def test_prompt_includes_context(self, sample_qa_pair, sample_distractors):
        """Prompt should include full context."""
        context = build_context(
            sample_qa_pair, sample_distractors, gold_position=5, total_docs=5, seed=42
        )

        prompt = build_prompt(context, "Test question?")

        # Check all documents are in prompt
        for i in range(1, 6):
            assert f"Document {i}:" in prompt

    def test_prompt_formatting(self, sample_qa_pair, sample_distractors):
        """Prompt should have proper formatting with instructions."""
        context = build_context(
            sample_qa_pair, sample_distractors, gold_position=1, total_docs=3, seed=42
        )

        prompt = build_prompt(context, "Question?")

        # Should have instruction
        assert (
            "Based on the following documents" in prompt
            or "documents" in prompt.lower()
        )
        # Should have question label
        assert "Question:" in prompt
        # Should have answer placeholder
        assert "Answer:" in prompt

    def test_empty_context(self, sample_qa_pair):
        """Should handle empty context string."""
        prompt = build_prompt("", sample_qa_pair["question"])

        assert isinstance(prompt, str)
        assert sample_qa_pair["question"] in prompt

    def test_special_characters_in_question(self, sample_qa_pair, sample_distractors):
        """Should handle questions with special characters."""
        context = build_context(
            sample_qa_pair, sample_distractors, gold_position=1, total_docs=3, seed=42
        )

        question = "What is 2+2? (multiplication: 2*2)"
        prompt = build_prompt(context, question)

        assert question in prompt


class TestIntegrationBuildContextPrompt:
    """Integration tests for build_context and build_prompt."""

    def test_full_pipeline(self, sample_qa_pair, sample_distractors):
        """Test full pipeline from QA to prompt."""
        context = build_context(
            sample_qa_pair, sample_distractors, gold_position=5, total_docs=10, seed=42
        )

        prompt = build_prompt(context, sample_qa_pair["question"])

        # Verify all pieces are in place
        assert sample_qa_pair["question"] in prompt
        assert sample_qa_pair["gold_doc"] in prompt
        assert "Document 5:" in prompt
        assert "Document 10:" in prompt
        assert len(prompt) > len(context)  # Prompt includes instructions
