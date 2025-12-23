# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Query expansion for LazyGraphRAG search."""

import json
import logging
from dataclasses import dataclass
from typing import Any

from graphrag.language_model.protocol.base import ChatModel
from graphrag.prompts.query.lazy_search_system_prompt import (
    LAZY_QUERY_EXPANSION_PROMPT,
)
from graphrag.query.llm.text_utils import try_parse_json_object

logger = logging.getLogger(__name__)


@dataclass
class QueryExpansionResult:
    """Result of query expansion.
    
    Attributes:
        original_query: The user's original query.
        subqueries: List of identified subqueries.
        expanded_query: Combined comprehensive query.
        reasoning: Explanation of expansion logic.
    """

    original_query: str
    subqueries: list[str]
    expanded_query: str
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "original_query": self.original_query,
            "subqueries": self.subqueries,
            "expanded_query": self.expanded_query,
            "reasoning": self.reasoning,
        }


class QueryExpander:
    """Expands user queries into subqueries for comprehensive search.
    
    Uses an LLM to identify implicit information needs in the user's query
    and generates subqueries that cover different aspects. The subqueries
    are then combined into a single expanded query.
    
    Attributes:
        model: The chat model for query expansion.
        max_subqueries: Maximum number of subqueries to generate.
        system_prompt: Custom system prompt (optional).
    """

    def __init__(
        self,
        model: ChatModel,
        max_subqueries: int = 5,
        system_prompt: str | None = None,
    ):
        """Initialize the QueryExpander.
        
        Args:
            model: ChatModel for LLM calls.
            max_subqueries: Maximum subqueries to generate (1-20).
            system_prompt: Custom prompt template (optional).
        """
        self.model = model
        self.max_subqueries = max_subqueries
        self.system_prompt = system_prompt or LAZY_QUERY_EXPANSION_PROMPT

    async def expand(
        self,
        query: str,
        **kwargs: Any,
    ) -> QueryExpansionResult:
        """Expand a query into subqueries and combined expanded query.
        
        Args:
            query: The user's original query.
            **kwargs: Additional parameters for LLM call.
            
        Returns:
            QueryExpansionResult containing subqueries and expanded query.
        """
        # Format the prompt
        prompt = self.system_prompt.format(
            query=query,
            max_subqueries=self.max_subqueries,
        )

        # Call the LLM
        try:
            response = await self.model.chat(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True,
                **kwargs,
            )
            
            # Extract content from response
            response_text = self._extract_response_text(response)
            
            # Parse the JSON response
            result = self._parse_response(response_text, query)
            
            logger.debug(
                "Query expanded: %d subqueries generated",
                len(result.subqueries),
            )
            
            return result
            
        except Exception as e:
            logger.warning("Query expansion failed: %s. Using original query.", str(e))
            # Fallback: return original query as-is
            return QueryExpansionResult(
                original_query=query,
                subqueries=[query],
                expanded_query=query,
                reasoning="Query expansion failed, using original query.",
            )

    def _extract_response_text(self, response: Any) -> str:
        """Extract text content from LLM response.
        
        Args:
            response: Raw LLM response.
            
        Returns:
            Text content as string.
        """
        if hasattr(response, "content"):
            return response.content
        if hasattr(response, "text"):
            return response.text
        if isinstance(response, dict):
            return response.get("content", response.get("text", str(response)))
        return str(response)

    def _parse_response(
        self,
        response_text: str,
        original_query: str,
    ) -> QueryExpansionResult:
        """Parse the LLM response into QueryExpansionResult.
        
        Args:
            response_text: Raw response text from LLM.
            original_query: The original user query.
            
        Returns:
            Parsed QueryExpansionResult.
        """
        try:
            # Try to parse JSON - returns (cleaned_str, parsed_dict)
            _, parsed = try_parse_json_object(response_text)
            
            if parsed:
                subqueries = parsed.get("subqueries", [original_query])
                expanded_query = parsed.get("expanded_query", original_query)
                reasoning = parsed.get("reasoning", "")
                
                # Limit subqueries to max
                subqueries = subqueries[: self.max_subqueries]
                
                # Ensure we have at least the original query
                if not subqueries:
                    subqueries = [original_query]
                if not expanded_query:
                    expanded_query = original_query
                
                return QueryExpansionResult(
                    original_query=original_query,
                    subqueries=subqueries,
                    expanded_query=expanded_query,
                    reasoning=reasoning,
                )
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response for query expansion")
        
        # Fallback
        return QueryExpansionResult(
            original_query=original_query,
            subqueries=[original_query],
            expanded_query=original_query,
            reasoning="Failed to parse expansion response.",
        )

    def expand_sync(self, query: str, **kwargs: Any) -> QueryExpansionResult:
        """Synchronous wrapper for expand().
        
        Args:
            query: The user's original query.
            **kwargs: Additional parameters for LLM call.
            
        Returns:
            QueryExpansionResult containing subqueries and expanded query.
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.expand(query, **kwargs)
        )
