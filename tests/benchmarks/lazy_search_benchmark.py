# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
Performance benchmarks for LazyGraphRAG search.

This script benchmarks LazySearch against other search methods to measure:
- Response time
- Token usage (budget consumption)
- Result quality metrics

Usage:
    python -m tests.benchmarks.lazy_search_benchmark
    
    # With specific preset
    python -m tests.benchmarks.lazy_search_benchmark --preset z500
    
    # Compare all presets
    python -m tests.benchmarks.lazy_search_benchmark --compare-presets
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from graphrag.config.models.lazy_search_config import LazySearchConfig
from graphrag.query.structured_search.lazy_search import (
    LazySearch,
    LazySearchData,
    LazySearchResult,
)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    
    method: str
    preset: str | None
    query: str
    response_length: int
    completion_time: float
    iterations_used: int
    chunks_processed: int
    budget_used: int
    claims_extracted: int
    relevant_sentences: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "preset": self.preset,
            "query": self.query[:50] + "..." if len(self.query) > 50 else self.query,
            "response_length": self.response_length,
            "completion_time": f"{self.completion_time:.3f}s",
            "iterations": self.iterations_used,
            "chunks": self.chunks_processed,
            "budget_used": self.budget_used,
            "claims": self.claims_extracted,
            "relevant": self.relevant_sentences,
        }


@dataclass
class BenchmarkSuite:
    """Suite of benchmark tests."""
    
    results: list[BenchmarkResult] = field(default_factory=list)
    
    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.results])
    
    def print_summary(self) -> None:
        """Print summary of benchmark results."""
        df = self.to_dataframe()
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        
        # Aggregate by preset
        if len(df) > 1:
            print("\nAVERAGES BY PRESET:")
            print("-" * 40)
            for preset in df["preset"].unique():
                subset = df[df["preset"] == preset]
                avg_time = subset["completion_time"].str.replace("s", "").astype(float).mean()
                avg_budget = subset["budget_used"].mean()
                print(f"  {preset}: avg_time={avg_time:.3f}s, avg_budget={avg_budget:.0f}")


def create_sample_data(num_chunks: int = 100) -> pd.DataFrame:
    """Create sample text chunks for benchmarking."""
    # Sample text about various topics
    topics = [
        "Machine learning is transforming industries through automation and intelligent systems.",
        "Deep learning neural networks can process complex patterns in large datasets.",
        "Natural language processing enables computers to understand human communication.",
        "Computer vision systems can identify objects, faces, and scenes in images.",
        "Reinforcement learning agents learn optimal strategies through trial and error.",
        "Transfer learning allows models to apply knowledge from one domain to another.",
        "Generative AI creates new content including text, images, and code.",
        "Federated learning enables training on distributed data without centralization.",
        "AutoML automates the process of selecting and tuning machine learning models.",
        "Edge AI brings machine learning inference to resource-constrained devices.",
    ]
    
    chunks = []
    for i in range(num_chunks):
        topic_idx = i % len(topics)
        chunks.append({
            "id": f"chunk_{i}",
            "text": f"{topics[topic_idx]} (Variation {i // len(topics) + 1})",
            "community_id": f"community_{topic_idx % 5}",
        })
    
    return pd.DataFrame(chunks)


class MockModelResponse:
    """Mock response object compatible with ModelResponse protocol."""
    
    def __init__(self, output_text: str):
        self.output = output_text
        self.content = output_text  # For _extract_response_text compatibility
        self.text = output_text     # Alternative access
    
    def __await__(self):
        """Make MockModelResponse awaitable."""
        async def _await():
            return self
        return _await().__await__()


class MockChatModel:
    """Mock chat model for benchmarking without API calls.
    
    This mock provides realistic responses for all LazySearch components:
    - Query expansion: Returns subqueries
    - Relevance testing: Returns relevance scores
    - Claim extraction: Returns extracted claims
    - Response generation: Returns synthesized answer
    """
    
    def __init__(self, response_delay: float = 0.01):
        self.response_delay = response_delay
        self.call_count = 0
        
        # Mock config for compatibility
        class MockConfig:
            model_name = "mock-model"
        self.config = MockConfig()
    
    def _detect_prompt_type(self, prompt: str) -> str:
        """Detect the type of prompt based on content."""
        prompt_lower = prompt.lower() if isinstance(prompt, str) else ""
        
        if "expand" in prompt_lower or "subquer" in prompt_lower:
            return "expansion"
        elif "relevance" in prompt_lower or "score" in prompt_lower:
            return "relevance"
        elif "claim" in prompt_lower or "extract" in prompt_lower:
            return "claims"
        else:
            return "response"
    
    def _generate_response(self, prompt_type: str) -> str:
        """Generate appropriate mock response based on prompt type."""
        responses = {
            "expansion": json.dumps({
                "subqueries": [
                    "What are the key concepts?",
                    "How does the system work?",
                    "What are the benefits?",
                ],
                "reasoning": "Expanded query into subtopics for comprehensive coverage."
            }),
            "relevance": json.dumps({
                "scores": [
                    {"sentence_index": 0, "score": 7, "reasoning": "Highly relevant"},
                    {"sentence_index": 1, "score": 8, "reasoning": "Very relevant"},
                    {"sentence_index": 2, "score": 4, "reasoning": "Somewhat relevant"},
                ]
            }),
            "claims": json.dumps({
                "claims": [
                    {
                        "statement": "The system uses advanced algorithms for processing.",
                        "confidence": 0.85,
                        "source_indices": [0, 1]
                    },
                    {
                        "statement": "Performance improvements are significant.",
                        "confidence": 0.9,
                        "source_indices": [1]
                    },
                ]
            }),
            "response": "Based on the provided context, the answer involves multiple aspects of the topic. The key points include advanced processing capabilities and improved performance metrics."
        }
        return responses.get(prompt_type, responses["response"])
    
    async def __call__(self, *args, **kwargs) -> str:
        """Simulate a chat completion."""
        self.call_count += 1
        await asyncio.sleep(self.response_delay)
        return "Mock response for benchmarking purposes."
    
    def chat(self, *args, **kwargs):
        """Sync chat method - returns awaitable MockModelResponse."""
        self.call_count += 1
        
        # Extract prompt from args or kwargs
        prompt = ""
        if args:
            prompt = str(args[0]) if args[0] else ""
        elif "messages" in kwargs:
            messages = kwargs["messages"]
            if messages and isinstance(messages, list):
                prompt = str(messages[-1].get("content", ""))
        elif "prompt" in kwargs:
            prompt = str(kwargs["prompt"])
        
        prompt_type = self._detect_prompt_type(prompt)
        response_text = self._generate_response(prompt_type)
        
        return MockModelResponse(response_text)
    
    async def achat(self, *args, **kwargs):
        """Async chat method."""
        self.call_count += 1
        await asyncio.sleep(self.response_delay)
        
        # Extract prompt from args or kwargs
        prompt = ""
        if args:
            prompt = str(args[0]) if args[0] else ""
        elif "messages" in kwargs:
            messages = kwargs["messages"]
            if messages and isinstance(messages, list):
                prompt = str(messages[-1].get("content", ""))
        elif "prompt" in kwargs:
            prompt = str(kwargs["prompt"])
        
        prompt_type = self._detect_prompt_type(prompt)
        response_text = self._generate_response(prompt_type)
        
        return MockModelResponse(response_text)
    
    def chat_stream(self, *args, **kwargs):
        """Sync streaming chat method."""
        self.call_count += 1
        response = self._generate_response("response")
        for chunk in response.split():
            yield chunk + " "
    
    async def achat_stream(self, *args, **kwargs):
        """Async streaming chat method."""
        self.call_count += 1
        await asyncio.sleep(self.response_delay)
        response = self._generate_response("response")
        for chunk in response.split():
            yield chunk + " "


async def benchmark_lazy_search(
    model: Any,
    data: LazySearchData,
    query: str,
    preset: str,
) -> BenchmarkResult:
    """Run a single benchmark for LazySearch."""
    config = LazySearchConfig.from_preset(preset)
    
    search = LazySearch(
        model=model,
        config=config,
        data=data,
    )
    
    start_time = time.perf_counter()
    result = await search.search(query)
    end_time = time.perf_counter()
    
    return BenchmarkResult(
        method="lazy_search",
        preset=preset,
        query=query,
        response_length=len(result.response),
        completion_time=end_time - start_time,
        iterations_used=result.iterations_used,
        chunks_processed=result.chunks_processed,
        budget_used=result.budget_used,
        claims_extracted=result.claims_extracted,
        relevant_sentences=result.relevant_sentences,
    )


async def run_benchmark_suite(
    queries: list[str],
    presets: list[str],
    num_chunks: int = 100,
) -> BenchmarkSuite:
    """Run full benchmark suite."""
    suite = BenchmarkSuite()
    
    # Create sample data
    print(f"Creating sample data with {num_chunks} chunks...")
    text_chunks = create_sample_data(num_chunks)
    data = LazySearchData(text_chunks=text_chunks)
    
    # Create mock model
    model = MockChatModel(response_delay=0.005)
    
    # Run benchmarks
    total_runs = len(queries) * len(presets)
    current_run = 0
    
    for preset in presets:
        for query in queries:
            current_run += 1
            print(f"  [{current_run}/{total_runs}] Running {preset} with query: {query[:30]}...")
            
            try:
                result = await benchmark_lazy_search(model, data, query, preset)
                suite.add_result(result)
            except Exception as e:
                print(f"    Error: {e}")
    
    return suite


def main():
    """Main entry point for benchmarks."""
    parser = argparse.ArgumentParser(description="LazyGraphRAG Performance Benchmarks")
    parser.add_argument(
        "--preset",
        choices=["z100", "z500", "z1500"],
        default=None,
        help="Specific preset to benchmark",
    )
    parser.add_argument(
        "--compare-presets",
        action="store_true",
        help="Compare all presets",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=100,
        help="Number of text chunks to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10,
        help="Number of queries to run (will generate variations)",
    )
    
    args = parser.parse_args()
    
    # Determine presets to test
    if args.compare_presets:
        presets = ["z100", "z500", "z1500"]
    elif args.preset:
        presets = [args.preset]
    else:
        presets = ["z500"]  # Default
    
    # Base queries for variation generation
    base_queries = [
        "What is machine learning and how does it work?",
        "Explain the difference between deep learning and traditional ML.",
        "How does natural language processing enable AI to understand text?",
        "What are the applications of computer vision in industry?",
        "Compare reinforcement learning with supervised learning approaches.",
        "What is transfer learning and why is it useful?",
        "How does generative AI create new content?",
        "What are the benefits of federated learning for privacy?",
        "Explain how AutoML simplifies model development.",
        "What is edge AI and what are its constraints?",
    ]
    
    # Generate query variations if more queries requested
    queries = []
    for i in range(args.num_queries):
        base_idx = i % len(base_queries)
        if i < len(base_queries):
            queries.append(base_queries[base_idx])
        else:
            variation = i // len(base_queries)
            queries.append(f"{base_queries[base_idx]} (variation {variation})")
    
    print("=" * 80)
    print("LazyGraphRAG Performance Benchmark")
    print("=" * 80)
    print(f"Presets: {presets}")
    print(f"Queries: {len(queries)}")
    print(f"Chunks: {args.num_chunks}")
    print("=" * 80)
    
    # Run benchmarks
    suite = asyncio.run(run_benchmark_suite(
        queries=queries,
        presets=presets,
        num_chunks=args.num_chunks,
    ))
    
    # Print results
    suite.print_summary()
    
    # Save to CSV if requested
    if args.output:
        df = suite.to_dataframe()
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    
    # Print preset comparison
    if args.compare_presets:
        print("\n" + "=" * 80)
        print("PRESET COMPARISON")
        print("=" * 80)
        print("""
        | Preset | Budget | Use Case                    | Cost    |
        |--------|--------|-----------------------------| --------|
        | z100   | 100    | Quick, low-cost queries     | Lowest  |
        | z500   | 500    | Balanced (recommended)      | Medium  |
        | z1500  | 1500   | High-quality, thorough      | Highest |
        
        LazyGraphRAG achieves comparable quality to full GraphRAG at ~1/100th cost.
        """)


if __name__ == "__main__":
    main()
