# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LazyGraphRAG search module.

Provides cost-efficient search implementation that achieves
comparable quality to full GraphRAG at ~1/100 of the cost
through iterative deepening and budget-controlled LLM calls.
"""

from graphrag.query.structured_search.lazy_search.claim_extractor import (
    ClaimExtractor,
)
from graphrag.query.structured_search.lazy_search.context import (
    LazyContextBuilder,
    LazySearchContext,
    merge_contexts,
)
from graphrag.query.structured_search.lazy_search.iterative_deepener import (
    DeepenerConfig,
    DeepeningResult,
    IterativeDeepener,
)
from graphrag.query.structured_search.lazy_search.query_expander import (
    QueryExpander,
    QueryExpansionResult,
)
from graphrag.query.structured_search.lazy_search.relevance_tester import (
    RelevanceTester,
    RelevanceTestResult,
)
from graphrag.query.structured_search.lazy_search.search import (
    LazySearch,
    LazySearchData,
    LazySearchResult,
)
from graphrag.query.structured_search.lazy_search.state import (
    Claim,
    LazySearchState,
    RelevantSentence,
)

__all__ = [
    # Main search
    "LazySearch",
    "LazySearchData",
    "LazySearchResult",
    # State management
    "Claim",
    "LazySearchState",
    "RelevantSentence",
    # Components
    "QueryExpander",
    "QueryExpansionResult",
    "RelevanceTester",
    "RelevanceTestResult",
    "ClaimExtractor",
    # Context
    "LazyContextBuilder",
    "LazySearchContext",
    "merge_contexts",
    # Iterative deepening
    "DeepenerConfig",
    "DeepeningResult",
    "IterativeDeepener",
]
