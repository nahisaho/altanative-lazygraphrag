# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Context building for LazyGraphRAG search."""

import logging
from dataclasses import dataclass
from typing import Any

import networkx as nx
import pandas as pd

from graphrag.query.structured_search.lazy_search.state import (
    Claim,
    RelevantSentence,
)

logger = logging.getLogger(__name__)


@dataclass
class LazySearchContext:
    """Container for search context used in response generation.
    
    Attributes:
        claims: Extracted claims relevant to query.
        relevant_sentences: Sentences with relevance scores.
        concept_graph: Subgraph of relevant concepts.
        formatted_context: Pre-formatted context string for LLM.
        total_tokens: Estimated token count.
    """
    claims: list[Claim]
    relevant_sentences: list[RelevantSentence]
    concept_graph: nx.Graph | None
    formatted_context: str
    total_tokens: int


class LazyContextBuilder:
    """Builds context for response generation in LazyGraphRAG.
    
    Takes extracted claims and relevant sentences and formats
    them into a context string suitable for the LLM to generate
    a response. Respects token budget constraints.
    
    Attributes:
        max_context_tokens: Maximum tokens allowed in context.
        include_sources: Whether to include source references.
        include_graph: Whether to include concept graph info.
    """

    def __init__(
        self,
        max_context_tokens: int = 8000,
        include_sources: bool = True,
        include_graph: bool = True,
        tokens_per_char: float = 0.25,  # Approximate
    ):
        """Initialize the LazyContextBuilder.
        
        Args:
            max_context_tokens: Maximum tokens for context.
            include_sources: Include source references in context.
            include_graph: Include concept graph summary.
            tokens_per_char: Estimated tokens per character.
        """
        self.max_context_tokens = max_context_tokens
        self.include_sources = include_sources
        self.include_graph = include_graph
        self.tokens_per_char = tokens_per_char

    def build_context(
        self,
        claims: list[Claim],
        relevant_sentences: list[RelevantSentence],
        concept_graph: nx.Graph | None = None,
        **kwargs: Any,
    ) -> LazySearchContext:
        """Build context for response generation.
        
        Formats claims and sentences into a coherent context
        string while respecting token limits.
        
        Args:
            claims: Extracted claims.
            relevant_sentences: Relevant sentences found.
            concept_graph: Optional concept subgraph.
            **kwargs: Additional parameters.
            
        Returns:
            LazySearchContext containing formatted context.
        """
        context_parts: list[str] = []
        current_tokens = 0
        
        # Add claims section
        if claims:
            claims_section = self._format_claims_section(claims)
            claims_tokens = self._estimate_tokens(claims_section)
            
            if current_tokens + claims_tokens <= self.max_context_tokens:
                context_parts.append(claims_section)
                current_tokens += claims_tokens
        
        # Add relevant sentences section
        if relevant_sentences:
            sentences_section = self._format_sentences_section(
                relevant_sentences,
                remaining_tokens=self.max_context_tokens - current_tokens,
            )
            sentences_tokens = self._estimate_tokens(sentences_section)
            
            if sentences_section and current_tokens + sentences_tokens <= self.max_context_tokens:
                context_parts.append(sentences_section)
                current_tokens += sentences_tokens
        
        # Add concept graph section
        if self.include_graph and concept_graph and concept_graph.number_of_nodes() > 0:
            graph_section = self._format_graph_section(concept_graph)
            graph_tokens = self._estimate_tokens(graph_section)
            
            if current_tokens + graph_tokens <= self.max_context_tokens:
                context_parts.append(graph_section)
                current_tokens += graph_tokens
        
        formatted_context = "\n\n".join(context_parts)
        
        return LazySearchContext(
            claims=claims,
            relevant_sentences=relevant_sentences,
            concept_graph=concept_graph,
            formatted_context=formatted_context,
            total_tokens=current_tokens,
        )

    def build_minimal_context(
        self,
        relevant_sentences: list[RelevantSentence],
        max_sentences: int = 20,
    ) -> str:
        """Build minimal context from sentences only.
        
        Used when claims extraction is skipped or failed.
        Provides a quick context for response generation.
        
        Args:
            relevant_sentences: Relevant sentences found.
            max_sentences: Maximum sentences to include.
            
        Returns:
            Formatted context string.
        """
        if not relevant_sentences:
            return "No relevant information found."
        
        # Sort by score and take top N
        sorted_sentences = sorted(
            relevant_sentences,
            key=lambda s: s.score,
            reverse=True,
        )[:max_sentences]
        
        context_lines = ["# Relevant Information\n"]
        
        for sentence in sorted_sentences:
            context_lines.append(f"- {sentence.text}")
        
        return "\n".join(context_lines)

    def _format_claims_section(self, claims: list[Claim]) -> str:
        """Format claims into a section string.
        
        Args:
            claims: List of claims to format.
            
        Returns:
            Formatted claims section.
        """
        lines = ["# Key Claims\n"]
        
        # Sort by confidence
        sorted_claims = sorted(claims, key=lambda c: c.confidence, reverse=True)
        
        for idx, claim in enumerate(sorted_claims, 1):
            confidence_str = f"[conf: {claim.confidence:.2f}]" if claim.confidence < 1.0 else ""
            lines.append(f"{idx}. {claim.statement} {confidence_str}")
            
            if self.include_sources and claim.source_chunk_ids:
                sources = ", ".join(claim.source_chunk_ids[:3])  # Max 3 sources
                lines.append(f"   Sources: {sources}")
        
        return "\n".join(lines)

    def _format_sentences_section(
        self,
        relevant_sentences: list[RelevantSentence],
        remaining_tokens: int,
    ) -> str:
        """Format relevant sentences into a section string.
        
        Args:
            relevant_sentences: List of sentences to format.
            remaining_tokens: Token budget remaining.
            
        Returns:
            Formatted sentences section.
        """
        lines = ["# Supporting Evidence\n"]
        
        # Sort by score
        sorted_sentences = sorted(
            relevant_sentences,
            key=lambda s: s.score,
            reverse=True,
        )
        
        current_tokens = self._estimate_tokens("\n".join(lines))
        
        for sentence in sorted_sentences:
            line = f"- [{sentence.chunk_id}] {sentence.text}"
            line_tokens = self._estimate_tokens(line)
            
            if current_tokens + line_tokens > remaining_tokens:
                break
            
            lines.append(line)
            current_tokens += line_tokens
        
        return "\n".join(lines)

    def _format_graph_section(self, graph: nx.Graph) -> str:
        """Format concept graph summary into a section string.
        
        Args:
            graph: NetworkX graph of concepts.
            
        Returns:
            Formatted graph section.
        """
        lines = ["# Related Concepts\n"]
        
        # Get top nodes by degree centrality
        if graph.number_of_nodes() > 0:
            centrality = nx.degree_centrality(graph)
            top_nodes = sorted(
                centrality.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            
            for node, score in top_nodes:
                neighbors = list(graph.neighbors(node))[:5]
                if neighbors:
                    neighbor_str = ", ".join(neighbors)
                    lines.append(f"- **{node}**: connected to {neighbor_str}")
                else:
                    lines.append(f"- **{node}**")
        
        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        Args:
            text: Input text.
            
        Returns:
            Estimated token count.
        """
        return int(len(text) * self.tokens_per_char)


def merge_contexts(
    contexts: list[LazySearchContext],
    max_total_tokens: int = 8000,
) -> LazySearchContext:
    """Merge multiple search contexts into one.
    
    Combines claims, sentences, and graphs from multiple
    search iterations into a unified context.
    
    Args:
        contexts: List of contexts to merge.
        max_total_tokens: Maximum tokens for merged context.
        
    Returns:
        Merged LazySearchContext.
    """
    if not contexts:
        return LazySearchContext(
            claims=[],
            relevant_sentences=[],
            concept_graph=None,
            formatted_context="",
            total_tokens=0,
        )
    
    if len(contexts) == 1:
        return contexts[0]
    
    # Merge claims (deduplicate by statement)
    all_claims: list[Claim] = []
    seen_statements: set[str] = set()
    
    for ctx in contexts:
        for claim in ctx.claims:
            if claim.statement not in seen_statements:
                all_claims.append(claim)
                seen_statements.add(claim.statement)
    
    # Merge sentences (deduplicate by text)
    all_sentences: list[RelevantSentence] = []
    seen_texts: set[str] = set()
    
    for ctx in contexts:
        for sentence in ctx.relevant_sentences:
            if sentence.text not in seen_texts:
                all_sentences.append(sentence)
                seen_texts.add(sentence.text)
    
    # Sort by score and limit
    all_sentences = sorted(
        all_sentences,
        key=lambda s: s.score,
        reverse=True,
    )[:100]  # Limit to top 100
    
    # Merge graphs
    merged_graph = nx.Graph()
    for ctx in contexts:
        if ctx.concept_graph:
            merged_graph = nx.compose(merged_graph, ctx.concept_graph)
    
    # Build merged formatted context
    builder = LazyContextBuilder(max_context_tokens=max_total_tokens)
    merged_context = builder.build_context(
        claims=all_claims,
        relevant_sentences=all_sentences,
        concept_graph=merged_graph if merged_graph.number_of_nodes() > 0 else None,
    )
    
    return merged_context
