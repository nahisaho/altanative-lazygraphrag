# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for LazySearch components."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from graphrag.query.structured_search.lazy_search.claim_extractor import (
    ClaimExtractor,
)
from graphrag.query.structured_search.lazy_search.context import (
    LazyContextBuilder,
    LazySearchContext,
    merge_contexts,
)
from graphrag.query.structured_search.lazy_search.query_expander import (
    QueryExpander,
    QueryExpansionResult,
)
from graphrag.query.structured_search.lazy_search.relevance_tester import (
    RelevanceTester,
    RelevanceTestResult,
)
from graphrag.query.structured_search.lazy_search.state import (
    Claim,
    LazySearchState,
    RelevantSentence,
)


def make_sentence(text: str, chunk_id: str, score: float = 0.9) -> RelevantSentence:
    """Helper to create RelevantSentence with required fields."""
    return RelevantSentence(
        text=text,
        score=score,
        chunk_id=chunk_id,
        community_id="test_community",
    )


class TestQueryExpander:
    """Tests for QueryExpander component."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock ChatModel."""
        model = MagicMock()
        model.chat = AsyncMock()
        return model

    @pytest.mark.asyncio
    async def test_expand_fallback_on_error(self, mock_model):
        """Test fallback to original query on error."""
        mock_model.chat.side_effect = Exception("LLM error")
        
        expander = QueryExpander(mock_model)
        result = await expander.expand("Test query")
        
        assert isinstance(result, QueryExpansionResult)
        assert result.subqueries == ["Test query"]

    @pytest.mark.asyncio
    async def test_expand_empty_response(self, mock_model):
        """Test handling of empty response."""
        mock_model.chat.return_value = MagicMock(content="{}")
        
        expander = QueryExpander(mock_model)
        result = await expander.expand("Test query")
        
        assert result.subqueries == ["Test query"]


class TestClaimExtractor:
    """Tests for ClaimExtractor component."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock ChatModel."""
        model = MagicMock()
        model.chat = AsyncMock()
        return model

    @pytest.fixture
    def sample_sentences(self):
        """Create sample relevant sentences."""
        return [
            make_sentence("GraphRAG is a knowledge graph system.", "c1", 0.9),
            make_sentence("It uses LLMs for extraction.", "c2", 0.8),
        ]

    @pytest.mark.asyncio
    async def test_extract_claims_success(self, mock_model, sample_sentences):
        """Test successful claim extraction."""
        # Create a proper mock response with content attribute
        mock_response = MagicMock()
        mock_response.content = '{"claims": [{"statement": "GraphRAG uses knowledge graphs", "confidence": 0.9, "source_indices": [0]}]}'
        mock_model.chat.return_value = mock_response
        
        extractor = ClaimExtractor(mock_model)
        claims = await extractor.extract_claims(
            relevant_sentences=sample_sentences,
            query="What is GraphRAG?",
        )
        
        assert len(claims) == 1
        assert "knowledge graphs" in claims[0].statement

    @pytest.mark.asyncio
    async def test_extract_claims_empty_input(self, mock_model):
        """Test extraction with no sentences."""
        extractor = ClaimExtractor(mock_model)
        claims = await extractor.extract_claims(
            relevant_sentences=[],
            query="Test",
        )
        
        assert claims == []
        mock_model.chat.assert_not_called()

    def test_deduplicate_claims(self, mock_model):
        """Test claim deduplication."""
        extractor = ClaimExtractor(mock_model)
        
        claims = [
            Claim(statement="GraphRAG uses graphs", source_chunk_ids=["c1"]),
            Claim(statement="GraphRAG uses knowledge graphs", source_chunk_ids=["c2"]),
            Claim(statement="LLMs are used for extraction", source_chunk_ids=["c3"]),
        ]
        
        deduplicated = extractor.deduplicate_claims(claims)
        
        # Similar claims should be merged
        assert len(deduplicated) <= len(claims)

    def test_text_similarity(self, mock_model):
        """Test text similarity calculation."""
        extractor = ClaimExtractor(mock_model)
        
        # Identical texts
        assert extractor._text_similarity("test text", "test text") == 1.0
        
        # Completely different
        sim = extractor._text_similarity("apple banana", "cat dog")
        assert sim == 0.0
        
        # Partial overlap
        sim = extractor._text_similarity("apple banana cherry", "apple banana date")
        assert 0 < sim < 1


class TestLazyContextBuilder:
    """Tests for LazyContextBuilder component."""

    @pytest.fixture
    def sample_claims(self):
        """Create sample claims."""
        return [
            Claim(statement="Claim 1", source_chunk_ids=["c1"], confidence=0.9),
            Claim(statement="Claim 2", source_chunk_ids=["c2"], confidence=0.8),
        ]

    @pytest.fixture
    def sample_sentences(self):
        """Create sample sentences."""
        return [
            make_sentence("Sentence 1", "c1", 0.9),
            make_sentence("Sentence 2", "c2", 0.7),
        ]

    def test_build_context(self, sample_claims, sample_sentences):
        """Test context building."""
        builder = LazyContextBuilder(max_context_tokens=8000)
        
        context = builder.build_context(
            claims=sample_claims,
            relevant_sentences=sample_sentences,
        )
        
        assert isinstance(context, LazySearchContext)
        assert len(context.claims) == 2
        assert len(context.relevant_sentences) == 2
        assert "Claim 1" in context.formatted_context

    def test_build_minimal_context(self, sample_sentences):
        """Test minimal context building."""
        builder = LazyContextBuilder()
        
        context = builder.build_minimal_context(
            relevant_sentences=sample_sentences,
            max_sentences=1,
        )
        
        assert isinstance(context, str)
        assert "Sentence 1" in context  # Highest score

    def test_token_estimation(self):
        """Test token count estimation."""
        builder = LazyContextBuilder(tokens_per_char=0.25)
        
        text = "This is a test string."  # 22 chars
        tokens = builder._estimate_tokens(text)
        
        assert tokens == int(22 * 0.25)

    def test_respects_token_limit(self, sample_claims):
        """Test context respects token limits."""
        # Very small limit
        builder = LazyContextBuilder(max_context_tokens=10)
        
        context = builder.build_context(
            claims=sample_claims,
            relevant_sentences=[],
        )
        
        # Should produce truncated context
        assert context.total_tokens <= 10 or context.formatted_context == ""


class TestMergeContexts:
    """Tests for merge_contexts function."""

    def test_merge_empty_list(self):
        """Test merging empty list."""
        result = merge_contexts([])
        
        assert isinstance(result, LazySearchContext)
        assert result.claims == []
        assert result.relevant_sentences == []

    def test_merge_single_context(self):
        """Test merging single context."""
        context = LazySearchContext(
            claims=[Claim(statement="Test", source_chunk_ids=["c1"])],
            relevant_sentences=[],
            concept_graph=None,
            formatted_context="Test context",
            total_tokens=10,
        )
        
        result = merge_contexts([context])
        
        assert len(result.claims) == 1

    def test_merge_multiple_contexts(self):
        """Test merging multiple contexts."""
        ctx1 = LazySearchContext(
            claims=[Claim(statement="Claim 1", source_chunk_ids=["c1"])],
            relevant_sentences=[make_sentence("S1", "c1", 0.9)],
            concept_graph=None,
            formatted_context="Context 1",
            total_tokens=10,
        )
        
        ctx2 = LazySearchContext(
            claims=[Claim(statement="Claim 2", source_chunk_ids=["c2"])],
            relevant_sentences=[make_sentence("S2", "c2", 0.8)],
            concept_graph=None,
            formatted_context="Context 2",
            total_tokens=10,
        )
        
        result = merge_contexts([ctx1, ctx2])
        
        assert len(result.claims) == 2
        assert len(result.relevant_sentences) == 2

    def test_merge_deduplicates(self):
        """Test that merging deduplicates claims and sentences."""
        claim = Claim(statement="Same claim", source_chunk_ids=["c1"])
        sentence = make_sentence("Same sentence", "c1", 0.9)
        
        ctx1 = LazySearchContext(
            claims=[claim],
            relevant_sentences=[sentence],
            concept_graph=None,
            formatted_context="",
            total_tokens=0,
        )
        
        ctx2 = LazySearchContext(
            claims=[claim],
            relevant_sentences=[sentence],
            concept_graph=None,
            formatted_context="",
            total_tokens=0,
        )
        
        result = merge_contexts([ctx1, ctx2])
        
        # Should deduplicate
        assert len(result.claims) == 1
        assert len(result.relevant_sentences) == 1
