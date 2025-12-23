# Technology Stack

**Project**: MS-GraphRAG+LazyGraphRAG
**Last Updated**: 2025-12-23
**Status**: 確定済み（既存GraphRAGコードベースに準拠）

---

## Overview

本プロジェクトは既存のMicrosoft GraphRAGコードベース（Python）に機能を追加するため、
既存の技術スタックに準拠します。

---

## Core Technology Stack

| Aspect | Technology | Version | Notes |
|--------|------------|---------|-------|
| **Primary Language** | Python | 3.10+ | 既存GraphRAGに準拠 |
| **Package Manager** | uv / pip | Latest | pyproject.toml |
| **Type Checking** | mypy | Latest | 静的型付け必須 |
| **Testing** | pytest | Latest | pytest-asyncio含む |
| **Async Runtime** | asyncio | Built-in | 非同期処理 |

---

## AI/ML Stack

| Component | Technology | Purpose |
|-----------|------------|----------|
| **LLM Interface** | ChatModel Protocol | LLMプロバイダー抽象化 |
| **Embeddings** | OpenAI / Azure OpenAI | ベクトル埋め込み |
| **NLP** | TextBlob / spaCy | 名詞句抽出（LazyGraphRAG） |
| **Graph** | NetworkX | グラフ操作 |
| **Vector Store** | Multiple Backends | ベクトル検索 |

---

## Data Processing

| Component | Technology | Purpose |
|-----------|------------|----------|
| **DataFrame** | pandas | データ操作 |
| **Numerical** | numpy | 数値計算 |
| **Tokenization** | tiktoken | トークンカウント |
| **Caching** | Pipeline Cache | 中間結果キャッシュ |

---

## Existing Query Implementations (Reference)

| Search Type | Location | Pattern |
|-------------|----------|----------|
| Local Search | `graphrag/query/structured_search/local_search/` | Entity-focused |
| Global Search | `graphrag/query/structured_search/global_search/` | Map-Reduce |
| DRIFT Search | `graphrag/query/structured_search/drift_search/` | Iterative refinement |
| **Lazy Search** | `graphrag/query/structured_search/lazy_search/` | **NEW: To be implemented** |

---

## Key Dependencies

```toml
[dependencies]
pandas = ">=2.0"
numpy = ">=1.24"
networkx = ">=3.0"
tiktoken = ">=0.5"
textblob = ">=0.17"  # LazyGraphRAG NLP extraction
aiohttp = ">=3.8"
pydantic = ">=2.0"
```

---

## Development Tools

| Tool | Purpose |
|------|----------|
| **Ruff** | Linting & Formatting |
| **mypy** | Static Type Checking |
| **pytest** | Unit & Integration Testing |
| **mkdocs** | Documentation |
| **pre-commit** | Git Hooks |

---

## Architecture Decisions

### ADR-001: BaseSearch継承
**Decision**: LazySearchはBaseSearchを継承し、既存のQueryインターフェースと互換性を維持
**Rationale**: 既存のコールバック、結果フォーマット、APIとの一貫性

### ADR-002: 状態管理クラス
**Decision**: LazySearchStateで探索状態を一元管理
**Rationale**: DRIFT Searchの QueryState パターンに準拠

### ADR-003: バッチLLM呼び出し
**Decision**: 関連性テストはバッチ処理でLLM呼び出し
**Rationale**: API呼び出し回数削減によるコスト・レイテンシ最適化

---

*既存GraphRAGコードベースとの一貫性を最優先としています。*
