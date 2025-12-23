# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""State management for LazyGraphRAG search."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RelevantSentence:
    """A sentence deemed relevant to the query.
    
    Attributes:
        text: The sentence text.
        score: Relevance score (0-10 scale).
        chunk_id: ID of the source text chunk.
        community_id: ID of the community containing this chunk.
        position_in_chunk: Position of sentence within the chunk.
    """

    text: str
    score: float
    chunk_id: str
    community_id: str
    position_in_chunk: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "score": self.score,
            "chunk_id": self.chunk_id,
            "community_id": self.community_id,
            "position_in_chunk": self.position_in_chunk,
        }


@dataclass
class Claim:
    """An extracted claim from relevant content.
    
    Attributes:
        statement: The claim statement text.
        source_chunk_ids: IDs of chunks supporting this claim.
        confidence: Confidence score (0-1).
        supporting_sentences: List of sentences supporting this claim.
    """

    statement: str
    source_chunk_ids: list[str]
    confidence: float = 1.0
    supporting_sentences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "statement": self.statement,
            "source_chunk_ids": self.source_chunk_ids,
            "confidence": self.confidence,
            "supporting_sentences": self.supporting_sentences,
        }


@dataclass
class LazySearchState:
    """State management for LazyGraphRAG search.
    
    Tracks query expansion, exploration progress, collected content,
    and budget consumption throughout the search process.
    
    Attributes:
        original_query: The user's original query.
        expanded_query: The LLM-expanded comprehensive query.
        subqueries: List of identified subqueries.
        visited_communities: Set of community IDs already explored.
        current_depth: Current depth in iterative deepening.
        consecutive_zero_relevance: Count of consecutive zero-relevance results.
        relevant_sentences: Collection of relevant sentences found.
        claims: Collection of extracted claims.
        budget_total: Total relevance test budget (e.g., 100, 500, 1500).
        budget_used: Number of relevance tests performed.
    """

    # Query State
    original_query: str = ""
    expanded_query: str = ""
    subqueries: list[str] = field(default_factory=list)

    # Exploration State
    visited_communities: set[str] = field(default_factory=set)
    current_depth: int = 0
    consecutive_zero_relevance: int = 0

    # Collection State
    relevant_sentences: list[RelevantSentence] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    processed_chunk_ids: set[str] = field(default_factory=set)

    # Budget State
    budget_total: int = 500
    budget_used: int = 0

    # Configuration thresholds
    zero_relevance_threshold: int = 3
    sufficient_relevance_count: int = 50
    max_depth: int = 3

    @property
    def budget_remaining(self) -> int:
        """Get remaining relevance test budget."""
        return max(0, self.budget_total - self.budget_used)

    @property
    def should_deepen(self) -> bool:
        """Check if search should deepen into subcommunities.
        
        Returns True when consecutive zero-relevance results exceed threshold.
        """
        return self.consecutive_zero_relevance >= self.zero_relevance_threshold

    @property
    def has_sufficient_content(self) -> bool:
        """Check if sufficient relevant content has been collected.
        
        Returns True when enough relevant sentences are found.
        """
        return len(self.relevant_sentences) >= self.sufficient_relevance_count

    @property
    def is_budget_exhausted(self) -> bool:
        """Check if relevance test budget is exhausted."""
        return self.budget_remaining <= 0

    @property
    def can_deepen_further(self) -> bool:
        """Check if depth limit allows further deepening."""
        return self.current_depth < self.max_depth

    @property
    def should_continue_exploration(self) -> bool:
        """Check if exploration should continue.
        
        Exploration continues if:
        - Budget is not exhausted
        - Sufficient content not yet collected
        """
        return not self.is_budget_exhausted and not self.has_sufficient_content

    def mark_community_visited(self, community_id: str) -> None:
        """Mark a community as visited."""
        self.visited_communities.add(community_id)

    def is_community_visited(self, community_id: str) -> bool:
        """Check if a community has been visited."""
        return community_id in self.visited_communities

    def add_relevant_sentence(self, sentence: RelevantSentence) -> None:
        """Add a relevant sentence to the collection."""
        self.relevant_sentences.append(sentence)

    def add_claim(self, claim: Claim) -> None:
        """Add a claim to the collection."""
        self.claims.append(claim)

    def consume_budget(self, amount: int = 1) -> None:
        """Consume relevance test budget.
        
        Args:
            amount: Number of budget units to consume.
        """
        self.budget_used += amount

    def record_relevance_result(self, found_relevant: bool) -> None:
        """Record a relevance test result for zero-relevance tracking.
        
        Args:
            found_relevant: Whether relevant content was found.
        """
        if found_relevant:
            self.consecutive_zero_relevance = 0
        else:
            self.consecutive_zero_relevance += 1

    def enter_subcommunity(self) -> None:
        """Enter a subcommunity (increase depth)."""
        self.current_depth += 1
        self.consecutive_zero_relevance = 0

    def exit_subcommunity(self) -> None:
        """Exit a subcommunity (decrease depth)."""
        if self.current_depth > 0:
            self.current_depth -= 1

    def reset_for_new_query(self) -> None:
        """Reset state for a new query while preserving configuration."""
        self.original_query = ""
        self.expanded_query = ""
        self.subqueries = []
        self.visited_communities = set()
        self.processed_chunk_ids = set()
        self.current_depth = 0
        self.consecutive_zero_relevance = 0
        self.relevant_sentences = []
        self.claims = []
        self.budget_used = 0

    @property
    def chunks_processed(self) -> set[str]:
        """Alias for processed_chunk_ids for compatibility."""
        return self.processed_chunk_ids

    def mark_chunk_processed(self, chunk_id: str) -> None:
        """Mark a chunk as processed."""
        self.processed_chunk_ids.add(chunk_id)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the current search state."""
        return {
            "original_query": self.original_query,
            "expanded_query": self.expanded_query,
            "subquery_count": len(self.subqueries),
            "communities_visited": len(self.visited_communities),
            "current_depth": self.current_depth,
            "relevant_sentences_count": len(self.relevant_sentences),
            "claims_count": len(self.claims),
            "budget_used": self.budget_used,
            "budget_total": self.budget_total,
            "budget_remaining": self.budget_remaining,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary representation."""
        return {
            "original_query": self.original_query,
            "expanded_query": self.expanded_query,
            "subqueries": self.subqueries,
            "visited_communities": list(self.visited_communities),
            "current_depth": self.current_depth,
            "consecutive_zero_relevance": self.consecutive_zero_relevance,
            "relevant_sentences": [s.to_dict() for s in self.relevant_sentences],
            "claims": [c.to_dict() for c in self.claims],
            "budget_total": self.budget_total,
            "budget_used": self.budget_used,
        }
