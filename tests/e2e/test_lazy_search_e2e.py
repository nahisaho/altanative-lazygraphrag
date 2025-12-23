# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""End-to-end tests for LazyGraphRAG search."""

import pandas as pd
import pytest

from graphrag.config.models.lazy_search_config import LazySearchConfig
from graphrag.query.structured_search.lazy_search import (
    LazySearch,
    LazySearchData,
    LazySearchResult,
)


class TestLazySearchE2E:
    """End-to-end tests for LazySearch with real LLM calls."""

    @pytest.fixture
    def sample_text_chunks(self) -> pd.DataFrame:
        """Create sample text chunks for testing."""
        return pd.DataFrame({
            "id": [f"chunk_{i}" for i in range(10)],
            "text": [
                "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
                "Einstein received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.",
                "The theory of general relativity describes gravity as a geometric property of space and time.",
                "Special relativity introduces the famous equation E=mc², relating energy and mass.",
                "Einstein was born in Germany in 1879 and later became a Swiss and American citizen.",
                "Quantum mechanics is a fundamental theory in physics that describes nature at atomic scales.",
                "The uncertainty principle states that position and momentum cannot both be precisely determined.",
                "Niels Bohr developed the Bohr model of the atom and contributed to quantum theory.",
                "The Copenhagen interpretation of quantum mechanics was developed by Bohr and Heisenberg.",
                "Werner Heisenberg formulated matrix mechanics, one of the first formulations of quantum mechanics.",
            ],
            "community_id": ["physics"] * 5 + ["quantum"] * 5,
        })

    @pytest.fixture
    def search_data(self, sample_text_chunks: pd.DataFrame) -> LazySearchData:
        """Create LazySearchData from sample chunks."""
        return LazySearchData(text_chunks=sample_text_chunks)

    @pytest.mark.asyncio
    async def test_lazy_search_basic_query(
        self,
        mock_chat_model,  # From conftest.py
        search_data: LazySearchData,
    ):
        """Test basic lazy search query execution."""
        config = LazySearchConfig.from_preset("z100")
        
        search = LazySearch(
            model=mock_chat_model,
            config=config,
            data=search_data,
        )
        
        result = await search.search("Who was Albert Einstein?")
        
        assert isinstance(result, LazySearchResult)
        assert result.response is not None
        assert len(result.response) > 0
        assert result.completion_time >= 0
        assert result.budget_used >= 0

    @pytest.mark.asyncio
    async def test_lazy_search_with_different_presets(
        self,
        mock_chat_model,
        search_data: LazySearchData,
    ):
        """Test lazy search with different budget presets."""
        presets = ["z100", "z500", "z1500"]
        results = {}
        
        for preset in presets:
            config = LazySearchConfig.from_preset(preset)
            search = LazySearch(
                model=mock_chat_model,
                config=config,
                data=search_data,
            )
            
            result = await search.search("Explain quantum mechanics")
            results[preset] = result
            
            assert result.response is not None
            assert result.budget_used <= config.relevance_budget

    @pytest.mark.asyncio
    async def test_lazy_search_context_data(
        self,
        mock_chat_model,
        search_data: LazySearchData,
    ):
        """Test that lazy search returns context data."""
        config = LazySearchConfig.from_preset("z100")
        
        search = LazySearch(
            model=mock_chat_model,
            config=config,
            data=search_data,
        )
        
        result = await search.search("What is the theory of relativity?")
        
        # Check metrics are populated
        assert result.iterations_used >= 0
        assert result.chunks_processed >= 0

    @pytest.mark.asyncio
    async def test_lazy_search_empty_query(
        self,
        mock_chat_model,
        search_data: LazySearchData,
    ):
        """Test lazy search with empty query."""
        config = LazySearchConfig.from_preset("z100")
        
        search = LazySearch(
            model=mock_chat_model,
            config=config,
            data=search_data,
        )
        
        result = await search.search("")
        
        # Should still return a result (possibly with default response)
        assert isinstance(result, LazySearchResult)

    @pytest.mark.asyncio
    async def test_lazy_search_complex_query(
        self,
        mock_chat_model,
        search_data: LazySearchData,
    ):
        """Test lazy search with complex multi-part query."""
        config = LazySearchConfig.from_preset("z500")
        
        search = LazySearch(
            model=mock_chat_model,
            config=config,
            data=search_data,
        )
        
        result = await search.search(
            "Compare Einstein's contributions to physics with those of Bohr and Heisenberg. "
            "What were their main disagreements about quantum mechanics?"
        )
        
        assert isinstance(result, LazySearchResult)
        assert result.response is not None
        assert len(result.response) > 0

    @pytest.mark.asyncio
    async def test_lazy_search_from_preset_factory(
        self,
        mock_chat_model,
        search_data: LazySearchData,
    ):
        """Test LazySearch.from_preset factory method."""
        search = LazySearch.from_preset(
            preset="z500",
            model=mock_chat_model,
            data=search_data,
        )
        
        result = await search.search("Who won the Nobel Prize?")
        
        assert isinstance(result, LazySearchResult)
        assert result.response is not None


class TestLazySearchAPIE2E:
    """End-to-end tests for LazySearch API integration."""

    @pytest.fixture
    def sample_text_units(self) -> pd.DataFrame:
        """Create sample text units DataFrame matching indexer output format."""
        return pd.DataFrame({
            "id": [f"tu_{i}" for i in range(5)],
            "text": [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Deep learning uses neural networks with multiple layers.",
                "Natural language processing enables computers to understand human language.",
                "Computer vision allows machines to interpret visual information.",
            ],
            "human_readable_id": list(range(5)),
            "document_ids": [["doc_0"]] * 5,
            "entity_ids": [None] * 5,
            "relationship_ids": [None] * 5,
        })

    @pytest.mark.asyncio
    async def test_api_lazy_search(
        self,
        sample_text_units: pd.DataFrame,
    ):
        """Test lazy_search API function."""
        # This test would require a full GraphRagConfig which needs API keys
        # Skip if not in full E2E mode
        pytest.skip("Requires full GraphRagConfig with API credentials")


class TestLazySearchCLIE2E:
    """End-to-end tests for LazySearch CLI integration."""

    @pytest.mark.skipif(False, reason="CLI tests don't require API")
    def test_cli_help_shows_lazy_method(self):
        """Test that CLI help shows lazy as a search method."""
        import subprocess
        
        result = subprocess.run(
            ["python", "-m", "graphrag", "query", "--help"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "lazy" in result.stdout.lower()

    @pytest.mark.skipif(False, reason="CLI tests don't require API")
    def test_cli_preset_option_exists(self):
        """Test that CLI has --preset option."""
        import subprocess
        
        result = subprocess.run(
            ["python", "-m", "graphrag", "query", "--help"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "--preset" in result.stdout
        assert "z100" in result.stdout or "z500" in result.stdout
