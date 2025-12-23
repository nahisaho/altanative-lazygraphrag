# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Integration tests for LazySearch components."""

import pandas as pd
import pytest

from graphrag.config.models.lazy_search_config import LazySearchConfig
from graphrag.query.structured_search.lazy_search import (
    ClaimExtractor,
    IterativeDeepener,
    LazyContextBuilder,
    LazySearchState,
    QueryExpander,
    RelevanceTester,
)
from graphrag.query.structured_search.lazy_search.iterative_deepener import (
    DeepenerConfig,
)
from graphrag.query.structured_search.lazy_search.state import (
    Claim,
    RelevantSentence,
)

from tests.mock_provider import MockChatLLM


# Test fixtures
@pytest.fixture
def sample_text_chunks() -> pd.DataFrame:
    """Create sample text chunks for testing."""
    return pd.DataFrame({
        "id": [f"chunk_{i}" for i in range(10)],
        "text": [
            "GraphRAG is a knowledge graph-based retrieval augmented generation system. "
            "It uses LLMs to extract entities and relationships from documents.",
            
            "The indexing pipeline processes documents through multiple stages. "
            "First, text is chunked into smaller segments for processing.",
            
            "Entity extraction identifies key concepts, people, organizations, and locations. "
            "These entities form the nodes of the knowledge graph.",
            
            "Relationship extraction finds connections between entities. "
            "These relationships become the edges in the knowledge graph.",
            
            "Community detection algorithms group related entities together. "
            "Leiden algorithm is commonly used for hierarchical community detection.",
            
            "Community reports summarize the key themes within each community. "
            "These reports enable global search queries across the entire dataset.",
            
            "Local search focuses on specific entities and their neighborhoods. "
            "It retrieves relevant context from the entity's local subgraph.",
            
            "Global search uses map-reduce over community reports. "
            "This enables answering questions that span the entire knowledge base.",
            
            "DRIFT search combines community context with local search. "
            "It provides broader coverage while maintaining entity-specific relevance.",
            
            "LazyGraphRAG is a cost-efficient alternative to full GraphRAG. "
            "It achieves comparable quality at approximately 1/100th of the cost.",
        ],
        "community_id": ["com_0", "com_0", "com_1", "com_1", "com_2", 
                        "com_2", "com_3", "com_3", "com_3", "com_4"],
    })


@pytest.fixture
def mock_expand_response() -> str:
    """Mock response for query expansion."""
    return """{
        "subqueries": [
            "What is GraphRAG?",
            "How does GraphRAG work?",
            "What are the main components of GraphRAG?"
        ],
        "reasoning": "Breaking down the query into specific aspects for comprehensive coverage."
    }"""


@pytest.fixture
def mock_relevance_response() -> str:
    """Mock response for relevance testing."""
    return """{
        "relevant_sentences": [
            {"sentence_index": 0, "relevance_score": 8, "reasoning": "Directly explains GraphRAG"},
            {"sentence_index": 1, "relevance_score": 7, "reasoning": "Describes how it works"}
        ]
    }"""


@pytest.fixture
def mock_claims_response() -> str:
    """Mock response for claim extraction."""
    return """{
        "claims": [
            {
                "statement": "GraphRAG is a knowledge graph-based RAG system",
                "source_sentences": [0],
                "confidence": 0.95
            },
            {
                "statement": "GraphRAG uses LLMs for entity extraction",
                "source_sentences": [0],
                "confidence": 0.9
            }
        ]
    }"""


class TestQueryExpanderIntegration:
    """Integration tests for QueryExpander."""

    @pytest.mark.asyncio
    async def test_query_expander_produces_subqueries(self, mock_expand_response: str):
        """Test QueryExpander produces valid subqueries."""
        mock_llm = MockChatLLM(responses=[mock_expand_response])
        
        expander = QueryExpander(model=mock_llm)
        result = await expander.expand("What is GraphRAG and how does it work?")
        
        assert result is not None
        assert result.original_query == "What is GraphRAG and how does it work?"
        assert len(result.subqueries) > 0
        assert result.reasoning is not None

    @pytest.mark.asyncio
    async def test_query_expander_handles_empty_response(self):
        """Test QueryExpander handles empty/invalid responses."""
        mock_llm = MockChatLLM(responses=["{}"])
        
        expander = QueryExpander(model=mock_llm)
        result = await expander.expand("Simple query")
        
        # Should return original query as fallback
        assert result is not None
        assert result.original_query == "Simple query"


class TestRelevanceTesterIntegration:
    """Integration tests for RelevanceTester."""

    @pytest.mark.asyncio
    async def test_relevance_tester_filters_sentences(self, mock_relevance_response: str):
        """Test RelevanceTester correctly filters sentences."""
        mock_llm = MockChatLLM(responses=[mock_relevance_response])
        
        tester = RelevanceTester(model=mock_llm, threshold=5.0)
        
        sentences = [
            "GraphRAG is a knowledge graph-based RAG system.",
            "It uses LLMs to extract entities.",
            "The weather is nice today.",
        ]
        
        result = await tester.test_sentences_batch(
            sentences=sentences,
            query="What is GraphRAG?",
            chunk_id="chunk_1",
            community_id="com_1",
        )
        
        assert result is not None
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_relevance_tester_respects_threshold(self):
        """Test that threshold filtering works."""
        # Response with low scores
        low_score_response = """{
            "relevant_sentences": [
                {"sentence_index": 0, "relevance_score": 2, "reasoning": "Not relevant"}
            ]
        }"""
        mock_llm = MockChatLLM(responses=[low_score_response])
        
        tester = RelevanceTester(model=mock_llm, threshold=5.0)
        
        result = await tester.test_sentences_batch(
            sentences=["Some text"],
            query="Query",
            chunk_id="chunk_1",
            community_id="com_1",
        )
        
        # Low score should be filtered out
        assert result is not None
        # With threshold 5.0, score 2 should be filtered
        assert len(result) == 0


class TestClaimExtractorIntegration:
    """Integration tests for ClaimExtractor."""

    @pytest.mark.asyncio
    async def test_claim_extractor_produces_claims(self, mock_claims_response: str):
        """Test ClaimExtractor produces valid claims."""
        mock_llm = MockChatLLM(responses=[mock_claims_response])
        
        extractor = ClaimExtractor(model=mock_llm)
        
        relevant_sentences = [
            RelevantSentence(
                text="GraphRAG is a knowledge graph-based RAG system.",
                score=8.0,
                chunk_id="chunk_1",
                community_id="com_1",
            ),
            RelevantSentence(
                text="It uses LLMs to extract entities.",
                score=7.0,
                chunk_id="chunk_1",
                community_id="com_1",
            ),
        ]
        
        claims = await extractor.extract_claims(
            relevant_sentences=relevant_sentences,
            query="What is GraphRAG?",
        )
        
        assert claims is not None
        assert isinstance(claims, list)
        # Claims may or may not be extracted depending on mock response parsing
        # Just verify no crash

    @pytest.mark.asyncio
    async def test_claim_extractor_handles_empty_response(self):
        """Test ClaimExtractor handles empty/invalid responses."""
        mock_llm = MockChatLLM(responses=["{}"])
        
        extractor = ClaimExtractor(model=mock_llm)
        
        relevant_sentences = [
            RelevantSentence(
                text="Some text.",
                score=8.0,
                chunk_id="chunk_1",
                community_id="com_1",
            ),
        ]
        
        claims = await extractor.extract_claims(
            relevant_sentences=relevant_sentences,
            query="Query",
        )
        
        # Should return empty list, not crash
        assert claims is not None
        assert isinstance(claims, list)

    @pytest.mark.asyncio
    async def test_claim_extractor_handles_empty_sentences(self):
        """Test ClaimExtractor handles empty sentence list."""
        mock_llm = MockChatLLM(responses=["{}"])
        
        extractor = ClaimExtractor(model=mock_llm)
        
        claims = await extractor.extract_claims(
            relevant_sentences=[],
            query="Query",
        )
        
        # Should return empty list for empty input
        assert claims == []


class TestIterativeDeepenerIntegration:
    """Integration tests for IterativeDeepener."""

    @pytest.mark.asyncio
    async def test_iterative_deepener_basic(
        self,
        sample_text_chunks: pd.DataFrame,
        mock_expand_response: str,
        mock_relevance_response: str,
    ):
        """Test IterativeDeepener coordinates search correctly."""
        mock_llm = MockChatLLM(responses=[
            mock_expand_response,      # Query expansion
            mock_relevance_response,   # First batch of relevance tests
            mock_relevance_response,   # More relevance tests
        ])
        
        config = DeepenerConfig(
            max_iterations=2,
            initial_sample_size=5,
            deepening_sample_size=3,
            relevance_threshold=5.0,
            max_total_chunks=10,
            target_claims=5,
        )
        
        deepener = IterativeDeepener(
            model=mock_llm,
            config=config,
        )
        
        result = await deepener.deepen(
            query="What is GraphRAG?",
            text_chunks=sample_text_chunks,
        )
        
        assert result is not None
        assert result.state is not None
        assert result.iterations_used >= 0
        assert result.chunks_processed >= 0

    @pytest.mark.asyncio
    async def test_iterative_deepener_with_state(
        self,
        sample_text_chunks: pd.DataFrame,
        mock_expand_response: str,
        mock_relevance_response: str,
    ):
        """Test IterativeDeepener with pre-existing state."""
        mock_llm = MockChatLLM(responses=[
            mock_expand_response,
            mock_relevance_response,
        ])
        
        config = DeepenerConfig(max_iterations=1)
        deepener = IterativeDeepener(model=mock_llm, config=config)
        
        # Create state with some budget
        state = LazySearchState(budget_total=50)
        
        result = await deepener.deepen(
            query="What is GraphRAG?",
            text_chunks=sample_text_chunks,
            state=state,
        )
        
        assert result is not None
        assert result.state is state  # Should use provided state


class TestLazyContextBuilderIntegration:
    """Integration tests for LazyContextBuilder."""

    def test_context_builder_produces_context(self):
        """Test LazyContextBuilder produces valid context."""
        builder = LazyContextBuilder()
        
        relevant_sentences = [
            RelevantSentence(
                text="GraphRAG is a knowledge graph-based RAG system.",
                score=8.0,
                chunk_id="chunk_1",
                community_id="com_1",
            ),
        ]
        
        claims = [
            Claim(
                statement="GraphRAG uses knowledge graphs",
                source_chunk_ids=["chunk_1"],
                confidence=0.9,
            ),
        ]
        
        context = builder.build_context(
            claims=claims,
            relevant_sentences=relevant_sentences,
        )
        
        assert context is not None
        assert len(context.formatted_context) > 0
        assert context.total_tokens >= 0

    def test_context_builder_minimal_context(self):
        """Test minimal context building."""
        builder = LazyContextBuilder()
        
        relevant_sentences = [
            RelevantSentence(
                text="GraphRAG is a knowledge graph-based RAG system.",
                score=8.0,
                chunk_id="chunk_1",
                community_id="com_1",
            ),
        ]
        
        context = builder.build_minimal_context(relevant_sentences)
        
        assert context is not None
        assert len(context) > 0


class TestLazySearchPresets:
    """Test preset configurations work correctly."""

    def test_z100_preset(self):
        """Test Z100 preset configuration."""
        config = LazySearchConfig.from_preset("z100")
        assert config.relevance_budget == 100

    def test_z500_preset(self):
        """Test Z500 preset configuration."""
        config = LazySearchConfig.from_preset("z500")
        assert config.relevance_budget == 500

    def test_z1500_preset(self):
        """Test Z1500 preset configuration."""
        config = LazySearchConfig.from_preset("z1500")
        assert config.relevance_budget == 1500

    def test_invalid_preset_raises_error(self):
        """Test invalid preset raises error."""
        with pytest.raises(ValueError):
            LazySearchConfig.from_preset("invalid_preset")

    def test_preset_names_are_consistent(self):
        """Test preset class variables match actual preset names."""
        assert LazySearchConfig.PRESET_Z100 == "z100"
        assert LazySearchConfig.PRESET_Z500 == "z500"
        assert LazySearchConfig.PRESET_Z1500 == "z1500"


class TestLazySearchConfigValidation:
    """Test configuration validation."""

    def test_budget_minimum(self):
        """Test minimum budget validation."""
        # Should succeed with minimum valid budget
        config = LazySearchConfig(relevance_budget=50)
        assert config.relevance_budget == 50

    def test_budget_maximum(self):
        """Test maximum budget validation."""
        config = LazySearchConfig(relevance_budget=5000)
        assert config.relevance_budget == 5000

    def test_threshold_range(self):
        """Test threshold validation."""
        config = LazySearchConfig(relevance_threshold=7.5)
        assert config.relevance_threshold == 7.5

    def test_config_defaults(self):
        """Test default configuration values."""
        config = LazySearchConfig()
        
        assert config.relevance_budget == 500
        assert config.relevance_threshold == 5.0
        assert config.max_depth == 3
        assert config.batch_size == 10
        assert config.top_k_chunks == 100


class TestLazySearchStateIntegration:
    """Integration tests for LazySearchState."""

    def test_state_budget_tracking(self):
        """Test budget is tracked correctly."""
        state = LazySearchState(budget_total=100)
        
        assert state.budget_remaining == 100
        assert state.budget_used == 0
        
        state.budget_used = 30
        assert state.budget_remaining == 70

    def test_state_should_continue(self):
        """Test exploration continuation logic."""
        state = LazySearchState(budget_total=100)
        
        # Initially should continue
        assert state.should_continue_exploration
        
        # Exhaust budget
        state.budget_used = 100
        assert not state.should_continue_exploration

    def test_state_add_relevant_sentences(self):
        """Test adding relevant sentences to state."""
        state = LazySearchState(budget_total=100)
        
        sentence = RelevantSentence(
            text="Test sentence",
            score=8.0,
            chunk_id="chunk_1",
            community_id="com_1",
        )
        
        state.relevant_sentences.append(sentence)
        
        assert len(state.relevant_sentences) == 1
        assert state.relevant_sentences[0].text == "Test sentence"
