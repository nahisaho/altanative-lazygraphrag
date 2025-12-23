# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for LazySearchState."""

import pytest

from graphrag.query.structured_search.lazy_search.state import (
    Claim,
    LazySearchState,
    RelevantSentence,
)


class TestRelevantSentence:
    """Tests for RelevantSentence dataclass."""

    def test_creation(self):
        """Test creating a RelevantSentence."""
        sentence = RelevantSentence(
            text="This is a test sentence.",
            score=0.8,
            chunk_id="chunk_001",
            community_id="community_001",
            position_in_chunk=0,
        )
        
        assert sentence.text == "This is a test sentence."
        assert sentence.chunk_id == "chunk_001"
        assert sentence.score == 0.8
        assert sentence.community_id == "community_001"

    def test_default_values(self):
        """Test default values for RelevantSentence."""
        sentence = RelevantSentence(
            text="Test",
            score=0.5,
            chunk_id="chunk_001",
            community_id="community_001",
        )
        
        assert sentence.position_in_chunk == 0

    def test_to_dict(self):
        """Test to_dict method."""
        sentence = RelevantSentence(
            text="Test",
            score=0.9,
            chunk_id="c1",
            community_id="com1",
        )
        
        data = sentence.to_dict()
        assert data["text"] == "Test"
        assert data["score"] == 0.9
        assert data["chunk_id"] == "c1"


class TestClaim:
    """Tests for Claim dataclass."""

    def test_creation(self):
        """Test creating a Claim."""
        claim = Claim(
            statement="The sky is blue.",
            source_chunk_ids=["chunk_001", "chunk_002"],
            confidence=0.95,
            supporting_sentences=["Evidence 1", "Evidence 2"],
        )
        
        assert claim.statement == "The sky is blue."
        assert len(claim.source_chunk_ids) == 2
        assert claim.confidence == 0.95
        assert len(claim.supporting_sentences) == 2

    def test_default_values(self):
        """Test default values for Claim."""
        claim = Claim(
            statement="Test claim",
            source_chunk_ids=["chunk_001"],
        )
        
        assert claim.confidence == 1.0
        assert claim.supporting_sentences == []


class TestLazySearchState:
    """Tests for LazySearchState."""

    def test_initialization(self):
        """Test state initialization."""
        state = LazySearchState(budget_total=100)
        
        assert state.budget_total == 100
        assert state.budget_used == 0
        assert state.subqueries == []
        assert state.relevant_sentences == []
        assert state.claims == []
        assert len(state.visited_communities) == 0

    def test_budget_remaining(self):
        """Test budget_remaining property."""
        state = LazySearchState(budget_total=100)
        state.budget_used = 30
        
        assert state.budget_remaining == 70

    def test_consume_budget(self):
        """Test consume_budget method."""
        state = LazySearchState(budget_total=100)
        
        state.consume_budget(25)
        assert state.budget_used == 25
        assert state.budget_remaining == 75
        
        state.consume_budget(50)
        assert state.budget_used == 75

    def test_is_budget_exhausted(self):
        """Test is_budget_exhausted property."""
        state = LazySearchState(budget_total=100)
        
        assert state.is_budget_exhausted is False
        
        state.budget_used = 100
        assert state.is_budget_exhausted is True

    def test_should_deepen(self):
        """Test should_deepen property."""
        state = LazySearchState(budget_total=100, zero_relevance_threshold=3)
        
        # No consecutive zeros yet
        assert state.should_deepen is False
        
        # Record multiple zero results
        state.record_relevance_result(False)
        state.record_relevance_result(False)
        assert state.should_deepen is False  # Only 2
        
        state.record_relevance_result(False)
        assert state.should_deepen is True  # Now 3

    def test_has_sufficient_content(self):
        """Test has_sufficient_content property."""
        state = LazySearchState(budget_total=100, sufficient_relevance_count=5)
        
        # No content
        assert state.has_sufficient_content is False
        
        # Add sentences
        for i in range(5):
            state.add_relevant_sentence(
                RelevantSentence(
                    text=f"Sentence {i}",
                    score=0.9,
                    chunk_id=f"c{i}",
                    community_id="com1",
                )
            )
        
        # Now sufficient
        assert state.has_sufficient_content is True

    def test_should_continue_exploration(self):
        """Test should_continue_exploration property."""
        state = LazySearchState(budget_total=100)
        
        # Has budget and no content, should continue
        assert state.should_continue_exploration is True
        
        # Exhaust budget
        state.budget_used = 100
        
        # No budget, should not continue
        assert state.should_continue_exploration is False

    def test_add_relevant_sentence(self):
        """Test adding relevant sentences."""
        state = LazySearchState(budget_total=100)
        
        sentence = RelevantSentence(
            text="Important finding",
            score=0.85,
            chunk_id="chunk_001",
            community_id="com1",
        )
        
        state.add_relevant_sentence(sentence)
        
        assert len(state.relevant_sentences) == 1
        assert state.relevant_sentences[0].text == "Important finding"

    def test_add_claim(self):
        """Test adding claims."""
        state = LazySearchState(budget_total=100)
        
        claim = Claim(
            statement="Key finding",
            source_chunk_ids=["chunk_001"],
            confidence=0.9,
        )
        
        state.add_claim(claim)
        
        assert len(state.claims) == 1
        assert state.claims[0].statement == "Key finding"

    def test_community_tracking(self):
        """Test community visit tracking."""
        state = LazySearchState(budget_total=100)
        
        assert state.is_community_visited("com1") is False
        
        state.mark_community_visited("com1")
        state.mark_community_visited("com2")
        state.mark_community_visited("com1")  # Duplicate
        
        assert len(state.visited_communities) == 2
        assert state.is_community_visited("com1") is True
        assert state.is_community_visited("com2") is True

    def test_depth_management(self):
        """Test depth management for iterative deepening."""
        state = LazySearchState(budget_total=100, max_depth=3)
        
        assert state.current_depth == 0
        assert state.can_deepen_further is True
        
        state.enter_subcommunity()
        assert state.current_depth == 1
        
        state.enter_subcommunity()
        state.enter_subcommunity()
        assert state.current_depth == 3
        assert state.can_deepen_further is False
        
        state.exit_subcommunity()
        assert state.current_depth == 2
        assert state.can_deepen_further is True

    def test_reset_for_new_query(self):
        """Test reset_for_new_query method."""
        state = LazySearchState(budget_total=100)
        
        # Add data
        state.budget_used = 50
        state.subqueries = ["q1", "q2"]
        state.original_query = "test query"
        state.add_relevant_sentence(
            RelevantSentence(text="Test", score=0.9, chunk_id="c1", community_id="com1")
        )
        state.add_claim(
            Claim(statement="Test", source_chunk_ids=["c1"])
        )
        state.mark_community_visited("com1")
        
        # Reset
        state.reset_for_new_query()
        
        # Verify reset
        assert state.budget_used == 0
        assert state.subqueries == []
        assert state.original_query == ""
        assert len(state.relevant_sentences) == 0
        assert len(state.claims) == 0
        assert len(state.visited_communities) == 0
        # budget_total should remain
        assert state.budget_total == 100

    def test_get_summary(self):
        """Test get_summary method."""
        state = LazySearchState(budget_total=100)
        state.original_query = "test"
        state.budget_used = 25
        
        summary = state.get_summary()
        
        assert summary["budget_total"] == 100
        assert summary["budget_used"] == 25
        assert summary["budget_remaining"] == 75
        assert summary["original_query"] == "test"
