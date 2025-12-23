# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Claim extraction for LazyGraphRAG search."""

import json
import logging
from typing import Any

import networkx as nx
import pandas as pd

from graphrag.language_model.protocol.base import ChatModel
from graphrag.prompts.query.lazy_search_system_prompt import (
    LAZY_CLAIM_EXTRACTION_PROMPT,
)
from graphrag.query.llm.text_utils import try_parse_json_object
from graphrag.query.structured_search.lazy_search.state import (
    Claim,
    RelevantSentence,
)

logger = logging.getLogger(__name__)


class ClaimExtractor:
    """Extracts claims from relevant content for LazyGraphRAG.
    
    Takes relevant sentences found during search and extracts
    factual claims that can be used to answer the query.
    Also handles deduplication of semantically similar claims.
    
    Attributes:
        model: The chat model for claim extraction.
        system_prompt: Custom system prompt (optional).
        deduplication_threshold: Similarity threshold for deduplication.
    """

    def __init__(
        self,
        model: ChatModel,
        system_prompt: str | None = None,
        deduplication_threshold: float = 0.85,
    ):
        """Initialize the ClaimExtractor.
        
        Args:
            model: ChatModel for LLM calls.
            system_prompt: Custom prompt template (optional).
            deduplication_threshold: Cosine similarity threshold for dedup.
        """
        self.model = model
        self.system_prompt = system_prompt or LAZY_CLAIM_EXTRACTION_PROMPT
        self.deduplication_threshold = deduplication_threshold

    async def extract_claims(
        self,
        relevant_sentences: list[RelevantSentence],
        query: str,
        **kwargs: Any,
    ) -> list[Claim]:
        """Extract claims from relevant sentences.
        
        Groups relevant sentences and uses LLM to extract
        factual claims that help answer the query.
        
        Args:
            relevant_sentences: List of relevant sentences found.
            query: The query being answered.
            **kwargs: Additional parameters for LLM call.
            
        Returns:
            List of extracted Claim objects.
        """
        if not relevant_sentences:
            return []
        
        # Format content for prompt
        content = self._format_content_for_prompt(relevant_sentences)
        
        # Format prompt
        prompt = self.system_prompt.format(
            query=query,
            content=content,
        )
        
        try:
            # Call LLM
            response = await self.model.chat(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True,
                **kwargs,
            )
            
            # Extract and parse response
            response_text = self._extract_response_text(response)
            claims = self._parse_claims_response(response_text, relevant_sentences)
            
            logger.debug("Extracted %d claims from %d sentences", 
                        len(claims), len(relevant_sentences))
            
            return claims
            
        except Exception as e:
            logger.warning("Claim extraction failed: %s", str(e))
            return []

    def deduplicate_claims(
        self,
        claims: list[Claim],
    ) -> list[Claim]:
        """Remove semantically duplicate claims.
        
        Uses simple text similarity to identify and merge
        duplicate claims, preserving source information.
        
        Args:
            claims: List of claims to deduplicate.
            
        Returns:
            Deduplicated list of claims.
        """
        if len(claims) <= 1:
            return claims
        
        # Simple deduplication based on text similarity
        deduplicated: list[Claim] = []
        seen_statements: list[str] = []
        
        for claim in claims:
            is_duplicate = False
            
            for seen in seen_statements:
                similarity = self._text_similarity(claim.statement, seen)
                if similarity >= self.deduplication_threshold:
                    is_duplicate = True
                    # Merge source information into existing claim
                    for existing in deduplicated:
                        if self._text_similarity(existing.statement, seen) >= self.deduplication_threshold:
                            existing.source_chunk_ids.extend(claim.source_chunk_ids)
                            existing.supporting_sentences.extend(claim.supporting_sentences)
                            break
                    break
            
            if not is_duplicate:
                deduplicated.append(claim)
                seen_statements.append(claim.statement)
        
        # Deduplicate source chunk IDs within each claim
        for claim in deduplicated:
            claim.source_chunk_ids = list(set(claim.source_chunk_ids))
            claim.supporting_sentences = list(set(claim.supporting_sentences))
        
        logger.debug(
            "Deduplicated claims: %d -> %d",
            len(claims),
            len(deduplicated),
        )
        
        return deduplicated

    def build_concept_subgraph(
        self,
        relevant_sentences: list[RelevantSentence],
        noun_graph_nodes: pd.DataFrame,
        noun_graph_edges: pd.DataFrame,
    ) -> nx.Graph:
        """Build concept subgraph from relevant sentences.
        
        Creates a subgraph containing only the concepts (noun phrases)
        that appear in the relevant sentences.
        
        Args:
            relevant_sentences: Relevant sentences found during search.
            noun_graph_nodes: Full noun graph nodes DataFrame.
            noun_graph_edges: Full noun graph edges DataFrame.
            
        Returns:
            NetworkX graph containing relevant concepts.
        """
        if relevant_sentences is None or len(relevant_sentences) == 0:
            return nx.Graph()
        
        # Get chunk IDs from relevant sentences
        chunk_ids = set(s.chunk_id for s in relevant_sentences)
        
        # Filter nodes that appear in these chunks
        if "text_unit_ids" in noun_graph_nodes.columns:
            relevant_nodes = noun_graph_nodes[
                noun_graph_nodes["text_unit_ids"].apply(
                    lambda ids: bool(set(ids) & chunk_ids) if isinstance(ids, list) else False
                )
            ]
        else:
            relevant_nodes = noun_graph_nodes
        
        # Get node titles
        node_titles = set(relevant_nodes["title"].tolist())
        
        # Filter edges
        relevant_edges = noun_graph_edges[
            (noun_graph_edges["source"].isin(node_titles)) &
            (noun_graph_edges["target"].isin(node_titles))
        ]
        
        # Build graph
        graph = nx.Graph()
        
        # Add nodes
        for _, row in relevant_nodes.iterrows():
            graph.add_node(
                row["title"],
                frequency=row.get("frequency", 1),
            )
        
        # Add edges
        for _, row in relevant_edges.iterrows():
            graph.add_edge(
                row["source"],
                row["target"],
                weight=row.get("weight", 1),
            )
        
        logger.debug(
            "Built concept subgraph: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        
        return graph

    def _format_content_for_prompt(
        self,
        relevant_sentences: list[RelevantSentence],
    ) -> str:
        """Format relevant sentences for the claim extraction prompt.
        
        Args:
            relevant_sentences: List of relevant sentences.
            
        Returns:
            Formatted string for prompt.
        """
        formatted = []
        for idx, sentence in enumerate(relevant_sentences):
            formatted.append(
                f"[{idx}] (chunk: {sentence.chunk_id}, score: {sentence.score:.1f}) "
                f"{sentence.text}"
            )
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

    def _parse_claims_response(
        self,
        response_text: str,
        relevant_sentences: list[RelevantSentence],
    ) -> list[Claim]:
        """Parse the LLM response into Claim objects.
        
        Args:
            response_text: Raw response text from LLM.
            relevant_sentences: Original sentences for reference.
            
        Returns:
            List of Claim objects.
        """
        claims: list[Claim] = []
        
        try:
            # try_parse_json_object returns (cleaned_str, parsed_dict)
            _, parsed = try_parse_json_object(response_text)
            
            if parsed and isinstance(parsed, dict):
                claims_data = parsed.get("claims", [])
                
                for item in claims_data:
                    statement = item.get("statement", "")
                    confidence = float(item.get("confidence", 1.0))
                    source_indices = item.get("source_indices", [])
                    
                    # Get source chunk IDs from indices
                    source_chunk_ids = []
                    supporting_sentences = []
                    
                    for idx in source_indices:
                        if idx < len(relevant_sentences):
                            source_chunk_ids.append(relevant_sentences[idx].chunk_id)
                            supporting_sentences.append(relevant_sentences[idx].text)
                    
                    if statement:
                        claims.append(
                            Claim(
                                statement=statement,
                                source_chunk_ids=list(set(source_chunk_ids)),
                                confidence=confidence,
                                supporting_sentences=supporting_sentences,
                            )
                        )
                        
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("Failed to parse claims response: %s", str(e))
        
        return claims

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity between two strings.
        
        Uses Jaccard similarity on word sets as a simple baseline.
        For production, consider using embeddings.
        
        Args:
            text1: First text string.
            text2: Second text string.
            
        Returns:
            Similarity score between 0 and 1.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
