# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Iterative deepening search for LazyGraphRAG.

Implements the core iterative deepening algorithm that:
1. Expands the query into subqueries
2. Tests text chunks for relevance (with budget constraints)
3. Extracts claims from relevant content
4. Deepens search on high-scoring chunks if needed
"""

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from graphrag.language_model.protocol.base import ChatModel
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
)
from graphrag.query.structured_search.lazy_search.relevance_tester import (
    RelevanceTester,
)
from graphrag.query.structured_search.lazy_search.state import (
    Claim,
    LazySearchState,
    RelevantSentence,
)

logger = logging.getLogger(__name__)


@dataclass
class DeepenerConfig:
    """Configuration for iterative deepening search.
    
    Attributes:
        max_iterations: Maximum deepening iterations.
        initial_sample_size: Initial chunks to sample.
        deepening_sample_size: Chunks to add each iteration.
        relevance_threshold: Minimum score to consider relevant.
        max_total_chunks: Maximum total chunks to process.
        target_claims: Target number of claims before stopping.
    """
    max_iterations: int = 3
    initial_sample_size: int = 50
    deepening_sample_size: int = 30
    relevance_threshold: float = 0.5
    max_total_chunks: int = 200
    target_claims: int = 10


@dataclass
class DeepeningResult:
    """Result of iterative deepening search.
    
    Attributes:
        context: Final search context.
        iterations_used: Number of iterations performed.
        chunks_processed: Total chunks processed.
        budget_remaining: Remaining LLM call budget.
        state: Final search state.
    """
    context: LazySearchContext
    iterations_used: int
    chunks_processed: int
    budget_remaining: int
    state: LazySearchState


class IterativeDeepener:
    """Performs iterative deepening search for LazyGraphRAG.
    
    Coordinates the search process by:
    1. Using QueryExpander to generate subqueries
    2. Using RelevanceTester to find relevant sentences
    3. Using ClaimExtractor to extract claims
    4. Using LazyContextBuilder to format context
    
    The search deepens iteratively until:
    - Budget is exhausted
    - Sufficient claims are found
    - Maximum iterations reached
    
    Attributes:
        model: ChatModel for LLM calls.
        query_expander: Expands queries into subqueries.
        relevance_tester: Tests relevance of text chunks.
        claim_extractor: Extracts claims from relevant content.
        context_builder: Builds context for response generation.
        config: Deepening configuration.
    """

    def __init__(
        self,
        model: ChatModel,
        query_expander: QueryExpander | None = None,
        relevance_tester: RelevanceTester | None = None,
        claim_extractor: ClaimExtractor | None = None,
        context_builder: LazyContextBuilder | None = None,
        config: DeepenerConfig | None = None,
    ):
        """Initialize the IterativeDeepener.
        
        Args:
            model: ChatModel for LLM calls.
            query_expander: Optional custom query expander.
            relevance_tester: Optional custom relevance tester.
            claim_extractor: Optional custom claim extractor.
            context_builder: Optional custom context builder.
            config: Optional deepening configuration.
        """
        self.model = model
        self.query_expander = query_expander or QueryExpander(model)
        self.relevance_tester = relevance_tester or RelevanceTester(model)
        self.claim_extractor = claim_extractor or ClaimExtractor(model)
        self.context_builder = context_builder or LazyContextBuilder()
        self.config = config or DeepenerConfig()

    async def deepen(
        self,
        query: str,
        text_chunks: pd.DataFrame,
        state: LazySearchState | None = None,
        noun_graph_nodes: pd.DataFrame | None = None,
        noun_graph_edges: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> DeepeningResult:
        """Perform iterative deepening search.
        
        Main entry point for the deepening algorithm.
        Processes chunks iteratively until stopping conditions met.
        
        Args:
            query: User's search query.
            text_chunks: DataFrame of text chunks to search.
            state: Optional existing search state.
            noun_graph_nodes: Optional noun graph nodes for concept subgraph.
            noun_graph_edges: Optional noun graph edges for concept subgraph.
            **kwargs: Additional parameters.
            
        Returns:
            DeepeningResult containing final context and metadata.
        """
        # Initialize state
        if state is None:
            state = LazySearchState(budget_total=100)
        
        # Expand query into subqueries
        expansion = await self.query_expander.expand(query, **kwargs)
        subqueries = expansion.subqueries if expansion.subqueries else [query]
        
        state.subqueries = subqueries
        logger.info(
            "Expanded query into %d subqueries, budget remaining: %d",
            len(subqueries),
            state.budget_remaining,
        )
        
        # Get initial sample of chunks
        chunk_ids = self._get_chunk_sample(
            text_chunks,
            size=self.config.initial_sample_size,
            exclude=state.chunks_processed,
        )
        
        iteration = 0
        contexts: list[LazySearchContext] = []
        
        while (
            iteration < self.config.max_iterations
            and state.should_continue_exploration
            and len(state.chunks_processed) < self.config.max_total_chunks
        ):
            iteration += 1
            logger.info(
                "Deepening iteration %d: processing %d chunks",
                iteration,
                len(chunk_ids),
            )
            
            # Process chunks for this iteration
            iteration_result = await self._process_iteration(
                query=query,
                subqueries=subqueries,
                chunk_ids=chunk_ids,
                text_chunks=text_chunks,
                state=state,
                noun_graph_nodes=noun_graph_nodes,
                noun_graph_edges=noun_graph_edges,
                **kwargs,
            )
            
            contexts.append(iteration_result)
            
            # Check stopping conditions
            if not state.should_continue_exploration:
                logger.info("Stopping: exploration criteria met")
                break
            
            if len(state.claims) >= self.config.target_claims:
                logger.info("Stopping: target claims reached")
                break
            
            # Get next sample of chunks (focusing on high-scoring areas)
            chunk_ids = self._get_deepening_sample(
                text_chunks=text_chunks,
                state=state,
                size=self.config.deepening_sample_size,
            )
            
            if not chunk_ids:
                logger.info("Stopping: no more chunks to process")
                break
        
        # Merge all iteration contexts
        final_context = merge_contexts(contexts)
        
        return DeepeningResult(
            context=final_context,
            iterations_used=iteration,
            chunks_processed=len(state.chunks_processed),
            budget_remaining=state.budget_remaining,
            state=state,
        )

    async def _process_iteration(
        self,
        query: str,
        subqueries: list[str],
        chunk_ids: list[str],
        text_chunks: pd.DataFrame,
        state: LazySearchState,
        noun_graph_nodes: pd.DataFrame | None = None,
        noun_graph_edges: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> LazySearchContext:
        """Process a single iteration of deepening.
        
        Tests chunks for relevance, extracts claims, and builds context.
        
        Args:
            query: Original query.
            subqueries: Expanded subqueries.
            chunk_ids: Chunk IDs to process in this iteration.
            text_chunks: Full DataFrame of text chunks.
            state: Current search state.
            noun_graph_nodes: Optional noun graph nodes.
            noun_graph_edges: Optional noun graph edges.
            **kwargs: Additional parameters.
            
        Returns:
            LazySearchContext for this iteration.
        """
        iteration_sentences: list[RelevantSentence] = []
        
        # Test each chunk for relevance
        for chunk_id in chunk_ids:
            if not state.budget_remaining:
                break
            
            # Get chunk data
            chunk_row = text_chunks[text_chunks["id"] == chunk_id]
            if chunk_row.empty:
                continue
            
            row = chunk_row.iloc[0]
            chunk_text = row.get("text", "")
            if not chunk_text:
                continue
            
            # Build chunk dict for test_chunk
            chunk = {
                "id": chunk_id,
                "text": chunk_text,
                "community_id": row.get("community_id", ""),
            }
            
            # Test chunk relevance using first subquery
            result = await self.relevance_tester.test_chunk(
                chunk=chunk,
                query=subqueries[0] if subqueries else query,
                state=state,
            )
            
            # Filter by relevance threshold
            for sentence in result:
                if sentence.score >= self.config.relevance_threshold:
                    iteration_sentences.append(sentence)
                    state.relevant_sentences.append(sentence)
            
            state.mark_chunk_processed(chunk_id)
        
        # Extract claims from relevant sentences
        if iteration_sentences:
            claims = await self.claim_extractor.extract_claims(
                relevant_sentences=iteration_sentences,
                query=query,
                **kwargs,
            )
            
            # Deduplicate and add to state
            claims = self.claim_extractor.deduplicate_claims(
                state.claims + claims
            )
            state.claims = claims
        
        # Build concept subgraph if graph data available
        concept_graph = None
        if noun_graph_nodes is not None and noun_graph_edges is not None:
            concept_graph = self.claim_extractor.build_concept_subgraph(
                relevant_sentences=iteration_sentences,
                noun_graph_nodes=noun_graph_nodes,
                noun_graph_edges=noun_graph_edges,
            )
        
        # Build context for this iteration
        context = self.context_builder.build_context(
            claims=state.claims,
            relevant_sentences=iteration_sentences,
            concept_graph=concept_graph,
        )
        
        logger.info(
            "Iteration complete: %d sentences, %d claims, %d tokens",
            len(iteration_sentences),
            len(state.claims),
            context.total_tokens,
        )
        
        return context

    def _get_chunk_sample(
        self,
        text_chunks: pd.DataFrame,
        size: int,
        exclude: set[str] | None = None,
    ) -> list[str]:
        """Get initial sample of chunks to process.
        
        Uses random sampling for the first iteration.
        
        Args:
            text_chunks: DataFrame of text chunks.
            size: Number of chunks to sample.
            exclude: Set of chunk IDs to exclude.
            
        Returns:
            List of chunk IDs to process.
        """
        exclude = exclude or set()
        
        # Get available chunks
        available = text_chunks[~text_chunks["id"].isin(exclude)]
        
        if len(available) <= size:
            return available["id"].tolist()
        
        # Random sample
        sample = available.sample(n=size)
        return sample["id"].tolist()

    def _get_deepening_sample(
        self,
        text_chunks: pd.DataFrame,
        state: LazySearchState,
        size: int,
    ) -> list[str]:
        """Get next sample of chunks for deepening.
        
        Focuses on chunks near high-scoring content
        (same document or adjacent chunks).
        
        Args:
            text_chunks: DataFrame of text chunks.
            state: Current search state with relevance info.
            size: Number of chunks to sample.
            
        Returns:
            List of chunk IDs to process.
        """
        # Get high-scoring chunk IDs
        high_scoring_chunks = {
            s.chunk_id for s in state.relevant_sentences
            if s.score >= self.config.relevance_threshold
        }
        
        if not high_scoring_chunks:
            # Fall back to random sampling
            return self._get_chunk_sample(
                text_chunks,
                size=size,
                exclude=state.chunks_processed,
            )
        
        # Find related chunks (same document_id if available)
        related_chunks: set[str] = set()
        
        if "document_id" in text_chunks.columns:
            high_scoring_docs = text_chunks[
                text_chunks["id"].isin(high_scoring_chunks)
            ]["document_id"].unique()
            
            related = text_chunks[
                (text_chunks["document_id"].isin(high_scoring_docs)) &
                (~text_chunks["id"].isin(state.chunks_processed))
            ]
            related_chunks = set(related["id"].tolist())
        
        # If not enough related chunks, add random ones
        if len(related_chunks) < size:
            additional = self._get_chunk_sample(
                text_chunks,
                size=size - len(related_chunks),
                exclude=state.chunks_processed | related_chunks,
            )
            related_chunks.update(additional)
        
        return list(related_chunks)[:size]
