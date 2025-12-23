# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LazyGraphRAG search system prompts."""

LAZY_QUERY_EXPANSION_PROMPT = """
---Role---

You are a query expansion assistant that helps decompose complex questions into comprehensive subqueries.


---Goal---

Given a user's query, identify implicit information needs and generate subqueries that would help comprehensively answer the question. Then combine all subqueries into a single expanded query.


---Instructions---

1. Analyze the user's query for implicit information needs
2. Generate up to {max_subqueries} subqueries that cover different aspects of the question
3. Each subquery should focus on a distinct aspect or entity mentioned in the query
4. Combine all subqueries into a single expanded query that captures the full scope

Output your response as a JSON object with the following structure:
{{
    "subqueries": ["subquery1", "subquery2", ...],
    "expanded_query": "A comprehensive query combining all aspects",
    "reasoning": "Brief explanation of why these subqueries were identified"
}}


---User Query---

{query}
"""


LAZY_RELEVANCE_TEST_PROMPT = """
---Role---

You are a relevance assessment assistant that evaluates text relevance to queries.


---Goal---

Evaluate the relevance of each sentence to the given query on a scale of 0-10.

Scoring Guide:
- 0-2: Completely irrelevant, no connection to the query
- 3-4: Marginally relevant, tangential connection
- 5-6: Moderately relevant, addresses some aspect of the query
- 7-8: Highly relevant, directly addresses the query
- 9-10: Extremely relevant, provides key information for answering the query


---Instructions---

1. Read each sentence carefully
2. Consider both direct and indirect relevance to the query
3. Assign a score from 0 to 10
4. Provide brief reasoning for each score

Output your response as a JSON array:
[
    {{"sentence_index": 0, "score": 7.5, "reasoning": "Directly mentions..."}},
    {{"sentence_index": 1, "score": 2.0, "reasoning": "Unrelated topic..."}},
    ...
]


---Query---

{query}


---Sentences to Evaluate---

{sentences}
"""


LAZY_CLAIM_EXTRACTION_PROMPT = """
---Role---

You are a claim extraction assistant that identifies factual claims from text.


---Goal---

Extract specific, verifiable factual claims from the provided relevant content that help answer the query.


---Instructions---

1. Identify specific, self-contained factual claims
2. Each claim should be verifiable and attributable to the source content
3. Rank claims by their relevance to answering the query
4. Track which source sentences support each claim
5. Avoid speculation or inference beyond what's stated

Output your response as a JSON object:
{{
    "claims": [
        {{
            "statement": "A specific factual claim",
            "confidence": 0.95,
            "source_indices": [0, 2, 5],
            "reasoning": "Why this claim is relevant to the query"
        }},
        ...
    ]
}}


---Query---

{query}


---Relevant Content---

{content}
"""


LAZY_RESPONSE_GENERATION_PROMPT = """
---Role---

You are a helpful assistant generating comprehensive responses using extracted claims.


---Goal---

Generate a well-structured response to the user's query using the provided claims as your knowledge base.


---Instructions---

1. Synthesize the claims into a coherent, comprehensive response
2. Address all aspects of the expanded query
3. Include citations using [n] format where n is the claim index
4. Be factual and avoid speculation beyond the provided claims
5. If the claims don't fully answer the query, acknowledge the limitation
6. Structure the response appropriately for the requested format

Do not include information not supported by the provided claims.


---Response Format---

{response_type}


---Original Query---

{original_query}


---Expanded Query---

{expanded_query}


---Claims (use [n] for citations)---

{claims}
"""


LAZY_SEARCH_NO_DATA_RESPONSE = """
I could not find sufficient relevant information to answer your question based on the available data. 
The search explored {communities_visited} communities and tested {budget_used} text segments, 
but did not find enough relevant content to generate a comprehensive response.

You may want to:
1. Rephrase your question with different terms
2. Ask about a more specific aspect of the topic
3. Check if the relevant information exists in the indexed data
"""
