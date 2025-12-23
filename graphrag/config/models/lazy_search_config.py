# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Configuration settings for LazyGraphRAG search."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field


class LazySearchConfig(BaseModel):
    """Configuration for LazyGraphRAG search.
    
    LazyGraphRAG uses deferred LLM evaluation with configurable
    relevance test budgets (Z100, Z500, Z1500) to balance cost and quality.
    
    Attributes:
        prompt: Custom query expansion prompt.
        relevance_test_prompt: Custom relevance test prompt.
        claim_extraction_prompt: Custom claim extraction prompt.
        response_prompt: Custom response generation prompt.
        chat_model_id: Model ID for LLM calls.
        embedding_model_id: Model ID for embeddings.
        max_subqueries: Maximum subqueries to generate.
        query_expansion_enabled: Whether to expand queries.
        top_k_chunks: Number of top chunks for initial retrieval.
        breadth_limit: Breadth limit for community exploration.
        relevance_budget: Relevance test budget (Z100, Z500, Z1500).
        relevance_threshold: Minimum relevance score (0-10).
        batch_size: Batch size for relevance testing.
        max_depth: Maximum depth for iterative deepening.
        zero_relevance_threshold: Consecutive zeros before deepening.
        sufficient_relevance_count: Target relevant sentences.
        response_type: Response format type.
        include_citations: Whether to include source citations.
        max_response_tokens: Maximum tokens for response.
        temperature: LLM temperature for generation.
        concurrency: Maximum concurrent LLM calls.
    """

    # Preset names as class variables
    PRESET_Z100: ClassVar[str] = "z100"
    PRESET_Z500: ClassVar[str] = "z500"
    PRESET_Z1500: ClassVar[str] = "z1500"

    # Prompts (customizable)
    prompt: str | None = Field(
        description="The query expansion prompt to use.",
        default=None,
    )
    relevance_test_prompt: str | None = Field(
        description="The relevance test prompt to use.",
        default=None,
    )
    claim_extraction_prompt: str | None = Field(
        description="The claim extraction prompt to use.",
        default=None,
    )
    response_prompt: str | None = Field(
        description="The response generation prompt to use.",
        default=None,
    )

    # Model configuration
    chat_model_id: str = Field(
        description="The model ID to use for lazy search.",
        default="default",
    )
    embedding_model_id: str = Field(
        description="The embedding model ID for similarity search.",
        default="default",
    )

    # Query Refinement Parameters
    max_subqueries: int = Field(
        description="Maximum number of subqueries to generate.",
        default=5,
        ge=1,
        le=20,
    )
    query_expansion_enabled: bool = Field(
        description="Whether to enable query expansion.",
        default=True,
    )

    # Query Matching Parameters
    top_k_chunks: int = Field(
        description="Number of top chunks to retrieve by embedding similarity.",
        default=100,
        ge=10,
        le=1000,
    )
    breadth_limit: int = Field(
        description="Breadth limit for community exploration.",
        default=10,
        ge=1,
        le=100,
    )

    # Relevance Testing Parameters
    relevance_budget: int = Field(
        description="Total budget for relevance tests (Z100, Z500, Z1500).",
        default=500,
        ge=50,
        le=5000,
    )
    relevance_threshold: float = Field(
        description="Minimum relevance score to consider relevant (0-10 scale).",
        default=5.0,
        ge=0.0,
        le=10.0,
    )
    batch_size: int = Field(
        description="Batch size for relevance testing LLM calls.",
        default=10,
        ge=1,
        le=50,
    )

    # Iterative Deepening Parameters
    max_depth: int = Field(
        description="Maximum depth for iterative deepening into subcommunities.",
        default=3,
        ge=1,
        le=10,
    )
    max_iterations: int = Field(
        description="Maximum iterations for iterative deepening.",
        default=5,
        ge=1,
        le=20,
    )
    initial_sample_size: int = Field(
        description="Number of chunks to sample in the first iteration.",
        default=20,
        ge=5,
        le=100,
    )
    deepening_sample_size: int = Field(
        description="Number of additional chunks per deepening iteration.",
        default=10,
        ge=1,
        le=50,
    )
    max_total_chunks: int = Field(
        description="Maximum total chunks to process across all iterations.",
        default=100,
        ge=10,
        le=1000,
    )
    target_claims: int = Field(
        description="Target number of claims to extract.",
        default=20,
        ge=5,
        le=100,
    )
    max_context_tokens: int = Field(
        description="Maximum tokens for context window.",
        default=8000,
        ge=1000,
        le=128000,
    )
    budget: int = Field(
        description="Alias for relevance_budget for backward compatibility.",
        default=500,
        ge=50,
        le=5000,
    )

    zero_relevance_threshold: int = Field(
        description="Consecutive zero-relevance results before deepening.",
        default=3,
        ge=1,
        le=20,
    )
    sufficient_relevance_count: int = Field(
        description="Target number of relevant sentences to collect.",
        default=50,
        ge=10,
        le=500,
    )

    # Response Generation Parameters
    response_type: str = Field(
        description="Type of response to generate.",
        default="multiple paragraphs",
    )
    include_citations: bool = Field(
        description="Whether to include source citations in response.",
        default=True,
    )
    max_response_tokens: int = Field(
        description="Maximum tokens for response generation.",
        default=2000,
        ge=100,
        le=16000,
    )

    # LLM Parameters
    temperature: float = Field(
        description="Temperature for LLM generation.",
        default=0.0,
        ge=0.0,
        le=2.0,
    )
    concurrency: int = Field(
        description="Maximum concurrent LLM calls.",
        default=10,
        ge=1,
        le=50,
    )

    @classmethod
    def from_preset(cls, preset: str) -> "LazySearchConfig":
        """Create a configuration from a preset name.
        
        Presets:
        - z100: Low budget (100 tests), faster but less thorough
        - z500: Medium budget (500 tests), balanced (default)
        - z1500: High budget (1500 tests), more thorough but slower
        
        Args:
            preset: Preset name (z100, z500, z1500).
            
        Returns:
            LazySearchConfig with preset values.
            
        Raises:
            ValueError: If preset name is unknown.
        """
        presets: dict[str, dict] = {
            cls.PRESET_Z100: {
                "relevance_budget": 100,
                "sufficient_relevance_count": 20,
                "top_k_chunks": 50,
                "breadth_limit": 5,
            },
            cls.PRESET_Z500: {
                "relevance_budget": 500,
                "sufficient_relevance_count": 50,
                "top_k_chunks": 100,
                "breadth_limit": 10,
            },
            cls.PRESET_Z1500: {
                "relevance_budget": 1500,
                "sufficient_relevance_count": 100,
                "top_k_chunks": 200,
                "breadth_limit": 20,
            },
        }

        preset_lower = preset.lower()
        if preset_lower not in presets:
            valid_presets = ", ".join(presets.keys())
            msg = f"Unknown preset '{preset}'. Valid presets: {valid_presets}"
            raise ValueError(msg)

        return cls(**presets[preset_lower])

    def get_preset_name(self) -> str | None:
        """Get the preset name if configuration matches a preset.
        
        Returns:
            Preset name if matches, None otherwise.
        """
        if self.relevance_budget <= 100:
            return self.PRESET_Z100
        elif self.relevance_budget <= 500:
            return self.PRESET_Z500
        elif self.relevance_budget <= 1500:
            return self.PRESET_Z1500
        return None
