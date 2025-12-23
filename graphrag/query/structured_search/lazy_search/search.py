# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LazyGraphRAG search implementation.

Main entry point for LazyGraphRAG search that provides
dramatically reduced cost (~1/100) while maintaining
comparable quality to full GraphRAG search.
"""

import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from graphrag.config.models.lazy_search_config import LazySearchConfig
from graphrag.language_model.protocol.base import ChatModel
from graphrag.prompts.query.lazy_search_system_prompt import (
    LAZY_RESPONSE_GENERATION_PROMPT,
)
from graphrag.query.context_builder.conversation_history import ConversationHistory
from graphrag.query.structured_search.base import SearchResult
from graphrag.query.structured_search.lazy_search.claim_extractor import (
    ClaimExtractor,
)
from graphrag.query.structured_search.lazy_search.context import (
    LazyContextBuilder,
    LazySearchContext,
)
from graphrag.query.structured_search.lazy_search.iterative_deepener import (
    DeepenerConfig,
    IterativeDeepener,
)
from graphrag.query.structured_search.lazy_search.query_expander import (
    QueryExpander,
)
from graphrag.query.structured_search.lazy_search.relevance_tester import (
    RelevanceTester,
)
from graphrag.query.structured_search.lazy_search.state import (
    LazySearchState,
)
from graphrag.tokenizer.get_tokenizer import get_tokenizer
from graphrag.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class LazySearchResult(SearchResult):
    """Extended search result with LazyGraphRAG-specific metadata.
    
    Attributes:
        iterations_used: Number of deepening iterations performed.
        chunks_processed: Total text chunks processed.
        budget_used: LLM call budget consumed.
        claims_extracted: Number of claims extracted.
        relevant_sentences: Number of relevant sentences found.
    """
    iterations_used: int = 0
    chunks_processed: int = 0
    budget_used: int = 0
    claims_extracted: int = 0
    relevant_sentences: int = 0


@dataclass
class LazySearchData:
    """Container for data required by LazySearch.
    
    Attributes:
        text_chunks: DataFrame of text chunks with 'id' and 'text' columns.
        noun_graph_nodes: Optional noun graph nodes DataFrame.
        noun_graph_edges: Optional noun graph edges DataFrame.
        entities: Optional entities DataFrame.
        relationships: Optional relationships DataFrame.
    """
    text_chunks: pd.DataFrame
    noun_graph_nodes: pd.DataFrame | None = None
    noun_graph_edges: pd.DataFrame | None = None
    entities: pd.DataFrame | None = None
    relationships: pd.DataFrame | None = None


class LazySearch:
    """LazyGraphRAG search implementation.
    
    Provides cost-efficient search (~1/100 cost of full GraphRAG)
    while maintaining comparable quality. Uses iterative deepening
    with budget-controlled LLM calls.
    
    Key features:
    - Budget-controlled LLM usage (Z100, Z500, Z1500 presets)
    - Iterative deepening with early stopping
    - Query expansion for comprehensive coverage
    - Claim extraction for structured responses
    
    Attributes:
        model: ChatModel for LLM calls.
        config: LazySearchConfig with search parameters.
        tokenizer: Tokenizer for token counting.
        data: LazySearchData containing text chunks and graphs.
    
    Example:
        ```python
        from graphrag.config.models.lazy_search_config import LazySearchConfig
        
        config = LazySearchConfig.from_preset("z500")
        search = LazySearch(
            model=chat_model,
            config=config,
            data=LazySearchData(text_chunks=chunks_df),
        )
        result = await search.search("What are the main themes?")
        ```
    """

    def __init__(
        self,
        model: ChatModel,
        config: LazySearchConfig,
        data: LazySearchData,
        tokenizer: Tokenizer | None = None,
        response_prompt: str | None = None,
    ):
        """Initialize LazySearch.
        
        Args:
            model: ChatModel for LLM calls.
            config: LazySearchConfig with search parameters.
            data: LazySearchData with text chunks and optional graphs.
            tokenizer: Optional custom tokenizer.
            response_prompt: Optional custom response generation prompt.
        """
        self.model = model
        self.config = config
        self.data = data
        self.tokenizer = tokenizer or get_tokenizer()
        self.response_prompt = response_prompt or LAZY_RESPONSE_GENERATION_PROMPT
        
        # Initialize components
        self._query_expander = QueryExpander(model)
        self._relevance_tester = RelevanceTester(
            model=model,
            threshold=config.relevance_threshold,
        )
        self._claim_extractor = ClaimExtractor(model)
        self._context_builder = LazyContextBuilder(
            max_context_tokens=config.max_context_tokens,
        )
        
        # Initialize deepener
        self._deepener = IterativeDeepener(
            model=model,
            query_expander=self._query_expander,
            relevance_tester=self._relevance_tester,
            claim_extractor=self._claim_extractor,
            context_builder=self._context_builder,
            config=DeepenerConfig(
                max_iterations=config.max_iterations,
                initial_sample_size=config.initial_sample_size,
                deepening_sample_size=config.deepening_sample_size,
                relevance_threshold=config.relevance_threshold,
                max_total_chunks=config.max_total_chunks,
                target_claims=config.target_claims,
            ),
        )
        
        # Track statistics
        self._llm_calls = 0
        self._prompt_tokens = 0
        self._output_tokens = 0

    async def search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs: Any,
    ) -> LazySearchResult:
        """Perform lazy search for the given query.
        
        Main entry point for search. Performs iterative deepening
        to find relevant content and generates a response.
        
        Args:
            query: User's search query.
            conversation_history: Optional conversation history.
            **kwargs: Additional parameters passed to components.
            
        Returns:
            LazySearchResult with response and metadata.
        """
        start_time = time.time()
        self._llm_calls = 0
        self._prompt_tokens = 0
        self._output_tokens = 0
        
        # Initialize search state
        state = LazySearchState(budget_total=self.config.budget)
        
        # Include conversation history in query if available
        full_query = self._build_query_with_history(query, conversation_history)
        
        # Perform iterative deepening
        deepening_result = await self._deepener.deepen(
            query=full_query,
            text_chunks=self.data.text_chunks,
            state=state,
            noun_graph_nodes=self.data.noun_graph_nodes,
            noun_graph_edges=self.data.noun_graph_edges,
            **kwargs,
        )
        
        # Generate response
        response = await self._generate_response(
            query=full_query,
            context=deepening_result.context,
            **kwargs,
        )
        
        completion_time = time.time() - start_time
        
        logger.info(
            "LazySearch completed in %.2fs: %d iterations, %d chunks, %d claims",
            completion_time,
            deepening_result.iterations_used,
            deepening_result.chunks_processed,
            len(deepening_result.state.claims),
        )
        
        return LazySearchResult(
            response=response,
            context_data=self._build_context_data(deepening_result.context),
            context_text=deepening_result.context.formatted_context,
            completion_time=completion_time,
            llm_calls=self._llm_calls,
            prompt_tokens=self._prompt_tokens,
            output_tokens=self._output_tokens,
            iterations_used=deepening_result.iterations_used,
            chunks_processed=deepening_result.chunks_processed,
            budget_used=self.config.budget - deepening_result.budget_remaining,
            claims_extracted=len(deepening_result.state.claims),
            relevant_sentences=len(deepening_result.state.relevant_sentences),
        )

    async def stream_search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream search results for the given query.
        
        Performs the same search as `search()` but streams
        the response generation.
        
        Args:
            query: User's search query.
            conversation_history: Optional conversation history.
            **kwargs: Additional parameters.
            
        Yields:
            Response text chunks as they are generated.
        """
        # Initialize search state
        state = LazySearchState(budget_total=self.config.budget)
        
        # Include conversation history in query if available
        full_query = self._build_query_with_history(query, conversation_history)
        
        # Perform iterative deepening (not streamed)
        deepening_result = await self._deepener.deepen(
            query=full_query,
            text_chunks=self.data.text_chunks,
            state=state,
            noun_graph_nodes=self.data.noun_graph_nodes,
            noun_graph_edges=self.data.noun_graph_edges,
            **kwargs,
        )
        
        # Stream response generation
        async for chunk in self._stream_response(
            query=full_query,
            context=deepening_result.context,
            **kwargs,
        ):
            yield chunk

    async def _generate_response(
        self,
        query: str,
        context: LazySearchContext,
        **kwargs: Any,
    ) -> str:
        """Generate response from context.
        
        Args:
            query: User's query.
            context: Search context with claims and sentences.
            **kwargs: Additional parameters.
            
        Returns:
            Generated response text.
        """
        if not context.formatted_context:
            return "申し訳ございませんが、関連する情報が見つかりませんでした。"
        
        prompt = self.response_prompt.format(
            query=query,
            context=context.formatted_context,
        )
        
        try:
            response = await self.model.chat(
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            
            self._llm_calls += 1
            
            if hasattr(response, "content"):
                return response.content
            if hasattr(response, "text"):
                return response.text
            return str(response)
            
        except Exception as e:
            logger.error("Response generation failed: %s", str(e))
            return f"回答生成中にエラーが発生しました: {str(e)}"

    async def _stream_response(
        self,
        query: str,
        context: LazySearchContext,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream response generation.
        
        Args:
            query: User's query.
            context: Search context.
            **kwargs: Additional parameters.
            
        Yields:
            Response text chunks.
        """
        if not context.formatted_context:
            yield "申し訳ございませんが、関連する情報が見つかりませんでした。"
            return
        
        prompt = self.response_prompt.format(
            query=query,
            context=context.formatted_context,
        )
        
        try:
            async for chunk in self.model.stream_chat(
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            ):
                if hasattr(chunk, "content"):
                    yield chunk.content
                elif hasattr(chunk, "text"):
                    yield chunk.text
                else:
                    yield str(chunk)
                    
        except Exception as e:
            logger.error("Stream response failed: %s", str(e))
            yield f"回答生成中にエラーが発生しました: {str(e)}"

    def _build_query_with_history(
        self,
        query: str,
        conversation_history: ConversationHistory | None,
    ) -> str:
        """Build query including conversation history.
        
        Args:
            query: Current query.
            conversation_history: Previous conversation.
            
        Returns:
            Query with history context.
        """
        if conversation_history is None or len(conversation_history.turns) == 0:
            return query
        
        history_text = conversation_history.to_string()
        return f"Previous conversation:\n{history_text}\n\nCurrent question: {query}"

    def _build_context_data(
        self,
        context: LazySearchContext,
    ) -> dict[str, pd.DataFrame]:
        """Build context data dictionary for result.
        
        Args:
            context: Search context.
            
        Returns:
            Dictionary of DataFrames with context data.
        """
        data: dict[str, pd.DataFrame] = {}
        
        # Claims DataFrame
        if context.claims:
            data["claims"] = pd.DataFrame([
                {
                    "statement": c.statement,
                    "confidence": c.confidence,
                    "sources": ", ".join(c.source_chunk_ids[:3]),
                }
                for c in context.claims
            ])
        
        # Sentences DataFrame
        if context.relevant_sentences:
            data["sentences"] = pd.DataFrame([
                {
                    "text": s.text,
                    "score": s.score,
                    "chunk_id": s.chunk_id,
                }
                for s in context.relevant_sentences
            ])
        
        return data

    @classmethod
    def from_preset(
        cls,
        preset: str,
        model: ChatModel,
        data: LazySearchData,
        **kwargs: Any,
    ) -> "LazySearch":
        """Create LazySearch from a preset configuration.
        
        Convenience method to create search with predefined
        budget configurations.
        
        Args:
            preset: Preset name ("z100", "z500", "z1500").
            model: ChatModel for LLM calls.
            data: LazySearchData with text chunks.
            **kwargs: Additional parameters for LazySearch.
            
        Returns:
            Configured LazySearch instance.
            
        Example:
            ```python
            search = LazySearch.from_preset(
                "z500",
                model=chat_model,
                data=data,
            )
            ```
        """
        config = LazySearchConfig.from_preset(preset)
        return cls(
            model=model,
            config=config,
            data=data,
            **kwargs,
        )
