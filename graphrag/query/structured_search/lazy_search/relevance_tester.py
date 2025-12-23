# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Relevance testing for LazyGraphRAG search."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from graphrag.language_model.protocol.base import ChatModel
from graphrag.prompts.query.lazy_search_system_prompt import (
    LAZY_RELEVANCE_TEST_PROMPT,
)
from graphrag.query.llm.text_utils import try_parse_json_object
from graphrag.query.structured_search.lazy_search.state import (
    LazySearchState,
    RelevantSentence,
)

logger = logging.getLogger(__name__)


@dataclass
class RelevanceTestResult:
    """Result of relevance testing for a single sentence.
    
    Attributes:
        sentence: The sentence text.
        score: Relevance score (0-10).
        is_relevant: Whether score exceeds threshold.
        reasoning: LLM's reasoning for the score.
    """

    sentence: str
    score: float
    is_relevant: bool
    reasoning: str = ""


class RelevanceTester:
    """Tests sentence-level relevance using LLM.
    
    Evaluates the relevance of text content to a query using an LLM.
    Supports batch processing to minimize API calls and budget tracking
    to control costs.
    
    Attributes:
        model: The chat model for relevance evaluation.
        threshold: Minimum score to consider relevant (0-10).
        batch_size: Number of sentences per LLM call.
        system_prompt: Custom system prompt (optional).
    """

    def __init__(
        self,
        model: ChatModel,
        threshold: float = 5.0,
        batch_size: int = 10,
        system_prompt: str | None = None,
    ):
        """Initialize the RelevanceTester.
        
        Args:
            model: ChatModel for LLM calls.
            threshold: Minimum relevance score (0-10, default 5.0).
            batch_size: Sentences per batch (default 10).
            system_prompt: Custom prompt template (optional).
        """
        self.model = model
        self.threshold = threshold
        self.batch_size = batch_size
        self.system_prompt = system_prompt or LAZY_RELEVANCE_TEST_PROMPT

    async def test_chunk(
        self,
        chunk: dict[str, Any],
        query: str,
        state: LazySearchState,
    ) -> list[RelevantSentence]:
        """Test all sentences in a chunk for relevance.
        
        Splits the chunk into sentences and tests them in batches,
        respecting the budget limit in state.
        
        Args:
            chunk: Text chunk dict with 'id', 'text', 'community_id' keys.
            query: The query to test relevance against.
            state: LazySearchState for budget tracking.
            
        Returns:
            List of RelevantSentence objects that passed threshold.
        """
        chunk_id = chunk.get("id", "")
        community_id = chunk.get("community_id", "")
        text = chunk.get("text", "")
        
        if not text:
            return []
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Limit to remaining budget
        sentences_to_test = sentences[: state.budget_remaining]
        
        if not sentences_to_test:
            logger.debug("Budget exhausted, skipping chunk %s", chunk_id)
            return []
        
        # Process in batches
        relevant_sentences: list[RelevantSentence] = []
        
        for i in range(0, len(sentences_to_test), self.batch_size):
            batch = sentences_to_test[i : i + self.batch_size]
            batch_start_idx = i
            
            # Test batch
            results = await self.test_sentences_batch(
                sentences=batch,
                query=query,
                chunk_id=chunk_id,
                community_id=community_id,
                start_position=batch_start_idx,
            )
            
            # Update budget
            state.consume_budget(len(batch))
            
            # Collect relevant sentences
            relevant_sentences.extend(results)
            
            # Check budget
            if state.is_budget_exhausted:
                logger.debug("Budget exhausted during batch processing")
                break
        
        return relevant_sentences

    async def test_sentences_batch(
        self,
        sentences: list[str],
        query: str,
        chunk_id: str,
        community_id: str,
        start_position: int = 0,
    ) -> list[RelevantSentence]:
        """Test a batch of sentences for relevance.
        
        Args:
            sentences: List of sentences to test.
            query: The query to test relevance against.
            chunk_id: ID of the source chunk.
            community_id: ID of the community.
            start_position: Starting position in original chunk.
            
        Returns:
            List of RelevantSentence objects that passed threshold.
        """
        if not sentences:
            return []
        
        # Format sentences for prompt
        sentences_text = self._format_sentences_for_prompt(sentences)
        
        # Format prompt
        prompt = self.system_prompt.format(
            query=query,
            sentences=sentences_text,
        )
        
        try:
            # Call LLM
            response = await self.model.chat(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True,
            )
            
            # Extract and parse response
            response_text = self._extract_response_text(response)
            results = self._parse_relevance_response(response_text, sentences)
            
            # Convert to RelevantSentence objects
            relevant = []
            for idx, result in enumerate(results):
                if result.is_relevant:
                    relevant.append(
                        RelevantSentence(
                            text=result.sentence,
                            score=result.score,
                            chunk_id=chunk_id,
                            community_id=community_id,
                            position_in_chunk=start_position + idx,
                        )
                    )
            
            return relevant
            
        except Exception as e:
            logger.warning("Relevance testing failed: %s", str(e))
            return []

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences.
        
        Uses a simple regex-based approach that handles common
        sentence boundaries.
        
        Args:
            text: Text to split.
            
        Returns:
            List of sentence strings.
        """
        # Simple sentence splitting regex
        # Handles ., !, ? followed by space or end of string
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
        
        # First clean up the text
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Split on sentence boundaries
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If no sentences found, treat whole text as one sentence
        if not sentences and text:
            sentences = [text]
        
        return sentences

    def _format_sentences_for_prompt(self, sentences: list[str]) -> str:
        """Format sentences for the LLM prompt.
        
        Args:
            sentences: List of sentences.
            
        Returns:
            Formatted string with numbered sentences.
        """
        formatted = []
        for idx, sentence in enumerate(sentences):
            formatted.append(f"[{idx}] {sentence}")
        return "\n".join(formatted)

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

    def _parse_relevance_response(
        self,
        response_text: str,
        sentences: list[str],
    ) -> list[RelevanceTestResult]:
        """Parse the LLM response into RelevanceTestResult objects.
        
        Args:
            response_text: Raw response text from LLM.
            sentences: Original sentences for reference.
            
        Returns:
            List of RelevanceTestResult objects.
        """
        results: list[RelevanceTestResult] = []
        
        try:
            # Try to parse JSON array - returns (cleaned_str, parsed_dict_or_list)
            _, parsed = try_parse_json_object(response_text)
            
            if parsed and isinstance(parsed, list):
                for item in parsed:
                    idx = item.get("sentence_index", 0)
                    score = float(item.get("score", 0))
                    reasoning = item.get("reasoning", "")
                    
                    # Get sentence text
                    sentence = sentences[idx] if idx < len(sentences) else ""
                    
                    results.append(
                        RelevanceTestResult(
                            sentence=sentence,
                            score=score,
                            is_relevant=score >= self.threshold,
                            reasoning=reasoning,
                        )
                    )
            else:
                # Handle case where response is a dict with array inside
                if isinstance(parsed, dict) and "results" in parsed:
                    return self._parse_relevance_response(
                        json.dumps(parsed["results"]),
                        sentences,
                    )
                    
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("Failed to parse relevance response: %s", str(e))
        
        # If parsing failed, return empty (conservative approach)
        if not results:
            logger.debug("No results parsed from relevance response")
            
        return results

    async def test_single_sentence(
        self,
        sentence: str,
        query: str,
        chunk_id: str = "",
        community_id: str = "",
    ) -> RelevantSentence | None:
        """Test a single sentence for relevance.
        
        Convenience method for testing individual sentences.
        
        Args:
            sentence: The sentence to test.
            query: The query to test relevance against.
            chunk_id: Optional chunk ID.
            community_id: Optional community ID.
            
        Returns:
            RelevantSentence if relevant, None otherwise.
        """
        results = await self.test_sentences_batch(
            sentences=[sentence],
            query=query,
            chunk_id=chunk_id,
            community_id=community_id,
        )
        return results[0] if results else None
