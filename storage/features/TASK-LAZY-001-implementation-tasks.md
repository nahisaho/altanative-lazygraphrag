# LazyGraphRAG 実装タスク分解

**Document ID**: TASK-LAZY-001
**Version**: 1.0
**Status**: Ready for Implementation
**Created**: 2025-12-23
**Requirements Reference**: REQ-LAZY-001
**Specification Reference**: SPEC-LAZY-001

---

## 概要

LazyGraphRAGクエリアルゴリズムの実装タスクを、依存関係と優先順位に基づいて分解します。

---

## Phase 1: 基盤モジュール（依存関係なし）

### TASK-LAZY-001-001: LazySearchState クラス実装
**Priority**: P0 (Critical Path)
**Estimated**: 2h
**File**: `graphrag/query/structured_search/lazy_search/state.py`

**要件カバレッジ**: REQ-LAZY-001-043

**実装内容**:
- [ ] `LazySearchState` dataclass
- [ ] `RelevantSentence` dataclass
- [ ] `Claim` dataclass
- [ ] プロパティ: `budget_remaining`, `should_deepen`, `has_sufficient_content`

**受入基準**:
- [ ] 全dataclassがPydanticまたはdataclassで実装
- [ ] 型ヒント完備
- [ ] ユニットテスト作成

---

### TASK-LAZY-001-002: LazySearchConfig 実装
**Priority**: P0 (Critical Path)
**Estimated**: 1h
**File**: `graphrag/config/models/lazy_search_config.py`

**要件カバレッジ**: REQ-LAZY-001-003, REQ-LAZY-001-021

**実装内容**:
- [ ] `LazySearchConfig` dataclass
- [ ] プリセット: Z100, Z500, Z1500
- [ ] `from_preset()` クラスメソッド
- [ ] 全パラメータのデフォルト値設定

**受入基準**:
- [ ] 既存の `*_config.py` パターンに準拠
- [ ] プリセット切り替えが動作

---

### TASK-LAZY-001-003: プロンプトテンプレート作成
**Priority**: P0 (Critical Path)
**Estimated**: 2h
**Files**: 
- `graphrag/prompts/query/lazy_query_expansion_prompt.py`
- `graphrag/prompts/query/lazy_relevance_test_prompt.py`
- `graphrag/prompts/query/lazy_claim_extraction_prompt.py`
- `graphrag/prompts/query/lazy_response_generation_prompt.py`

**要件カバレッジ**: REQ-LAZY-001-NF-003

**実装内容**:
- [ ] クエリ展開プロンプト（JSON出力）
- [ ] 関連性テストプロンプト（バッチ対応、0-10スコア）
- [ ] 主張抽出プロンプト（出典追跡）
- [ ] 応答生成プロンプト（引用形式）

**受入基準**:
- [ ] 既存プロンプト形式に準拠
- [ ] 変数プレースホルダー使用

---

## Phase 2: コンポーネント実装（Phase 1依存）

### TASK-LAZY-001-010: QueryExpander 実装
**Priority**: P1
**Estimated**: 3h
**File**: `graphrag/query/structured_search/lazy_search/query_expander.py`
**Depends On**: TASK-LAZY-001-003

**要件カバレッジ**: REQ-LAZY-001-001, REQ-LAZY-001-002, REQ-LAZY-001-003

**実装内容**:
- [ ] `QueryExpander` クラス
- [ ] `expand()` 非同期メソッド
- [ ] `QueryExpansionResult` dataclass
- [ ] JSON応答パース処理

**受入基準**:
- [ ] サブクエリが正しく抽出される
- [ ] 展開クエリが元のクエリの意図を保持
- [ ] max_subqueries制限が機能

---

### TASK-LAZY-001-011: RelevanceTester 実装
**Priority**: P1
**Estimated**: 4h
**File**: `graphrag/query/structured_search/lazy_search/relevance_tester.py`
**Depends On**: TASK-LAZY-001-001, TASK-LAZY-001-003

**要件カバレッジ**: REQ-LAZY-001-020, REQ-LAZY-001-021, REQ-LAZY-001-022, REQ-LAZY-001-023

**実装内容**:
- [ ] `RelevanceTester` クラス
- [ ] `test_chunk()` 非同期メソッド
- [ ] `test_sentences_batch()` 非同期メソッド
- [ ] 文分割ロジック
- [ ] 予算追跡・制限処理
- [ ] `RelevanceTestResult` dataclass

**受入基準**:
- [ ] バッチ処理が正しく動作
- [ ] 予算超過時に適切に停止
- [ ] スコア閾値フィルタリングが機能

---

### TASK-LAZY-001-012: ClaimExtractor 実装
**Priority**: P1
**Estimated**: 3h
**File**: `graphrag/query/structured_search/lazy_search/claim_extractor.py`
**Depends On**: TASK-LAZY-001-001, TASK-LAZY-001-003

**要件カバレッジ**: REQ-LAZY-001-060, REQ-LAZY-001-061, REQ-LAZY-001-062

**実装内容**:
- [ ] `ClaimExtractor` クラス
- [ ] `extract_claims()` 非同期メソッド
- [ ] `deduplicate_claims()` メソッド（意味的重複除去）
- [ ] `build_concept_subgraph()` メソッド

**受入基準**:
- [ ] 主張が正しく抽出される
- [ ] 出典情報が保持される
- [ ] 重複除去が機能

---

### TASK-LAZY-001-013: LazyContextBuilder 実装
**Priority**: P1
**Estimated**: 4h
**File**: `graphrag/query/structured_search/lazy_search/context.py`
**Depends On**: TASK-LAZY-001-001

**要件カバレッジ**: REQ-LAZY-001-010, REQ-LAZY-001-011, REQ-LAZY-001-012

**実装内容**:
- [ ] `LazyContextBuilder` クラス
- [ ] `get_top_k_chunks()` 非同期メソッド（埋め込み類似度）
- [ ] `get_community_for_chunk()` メソッド
- [ ] `get_subcommunities()` メソッド
- [ ] `get_chunks_in_community()` メソッド
- [ ] `rank_communities_by_relevance()` メソッド

**受入基準**:
- [ ] 既存のContextBuilderパターンに準拠
- [ ] NounGraphデータを正しく読み込み
- [ ] コミュニティ階層を正しくナビゲート

---

### TASK-LAZY-001-014: IterativeDeepener 実装
**Priority**: P1
**Estimated**: 4h
**File**: `graphrag/query/structured_search/lazy_search/iterative_deepener.py`
**Depends On**: TASK-LAZY-001-011, TASK-LAZY-001-013

**要件カバレッジ**: REQ-LAZY-001-040, REQ-LAZY-001-041, REQ-LAZY-001-042

**実装内容**:
- [ ] `IterativeDeepener` クラス
- [ ] `explore()` 非同期メソッド
- [ ] `_explore_community()` 内部メソッド
- [ ] `_should_continue()` 判定メソッド
- [ ] 再帰的サブコミュニティ探索ロジック

**受入基準**:
- [ ] ゼロ関連性検出で適切に深化
- [ ] 最大深度制限が機能
- [ ] 早期終了条件が機能

---

## Phase 3: 統合実装（Phase 2依存）

### TASK-LAZY-001-020: LazySearch メインクラス実装
**Priority**: P0 (Critical Path)
**Estimated**: 6h
**File**: `graphrag/query/structured_search/lazy_search/search.py`
**Depends On**: TASK-LAZY-001-010, TASK-LAZY-001-011, TASK-LAZY-001-012, TASK-LAZY-001-013, TASK-LAZY-001-014

**要件カバレッジ**: REQ-LAZY-001-080, REQ-LAZY-001-081, REQ-LAZY-001-082, REQ-LAZY-001-NF-004

**実装内容**:
- [ ] `LazySearch` クラス（BaseSearch継承）
- [ ] `LazySearchResult` dataclass
- [ ] `search()` 非同期メソッド
- [ ] `stream_search()` 非同期ジェネレータ
- [ ] 全コンポーネント統合
- [ ] コールバック呼び出し
- [ ] LLM統計追跡

**受入基準**:
- [ ] E2E検索フローが動作
- [ ] SearchResult形式で結果返却
- [ ] ストリーミング応答が機能
- [ ] 既存コールバックと互換

---

### TASK-LAZY-001-021: __init__.py とパッケージエクスポート
**Priority**: P1
**Estimated**: 1h
**File**: `graphrag/query/structured_search/lazy_search/__init__.py`
**Depends On**: TASK-LAZY-001-020

**実装内容**:
- [ ] 公開API定義
- [ ] `__all__` リスト
- [ ] 型エクスポート

**受入基準**:
- [ ] 他のsearch実装と同じパターン

---

## Phase 4: 統合・テスト

### TASK-LAZY-001-030: ユニットテスト作成
**Priority**: P1
**Estimated**: 6h
**Files**: `tests/unit/query/structured_search/lazy_search/`
**Depends On**: Phase 3完了

**実装内容**:
- [ ] `test_state.py` - LazySearchState テスト
- [ ] `test_query_expander.py` - QueryExpander テスト
- [ ] `test_relevance_tester.py` - RelevanceTester テスト
- [ ] `test_claim_extractor.py` - ClaimExtractor テスト
- [ ] `test_iterative_deepener.py` - IterativeDeepener テスト
- [ ] `test_search.py` - LazySearch テスト

**受入基準**:
- [ ] カバレッジ80%以上
- [ ] 全テストパス
- [ ] モック使用は最小限

---

### TASK-LAZY-001-031: 統合テスト作成
**Priority**: P1
**Estimated**: 4h
**File**: `tests/integration/query/lazy_search/test_lazy_search_e2e.py`
**Depends On**: TASK-LAZY-001-030

**実装内容**:
- [ ] E2E検索テスト（実LLM使用）
- [ ] プリセット別テスト（Z100, Z500, Z1500）
- [ ] ストリーミングテスト

**受入基準**:
- [ ] 実サービス使用（Article IX準拠）
- [ ] 全プリセットでテスト合格

---

### TASK-LAZY-001-032: CLI/API統合
**Priority**: P2
**Estimated**: 3h
**Files**: 
- `graphrag/cli/query.py` (修正)
- `graphrag/api/query.py` (修正)
**Depends On**: TASK-LAZY-001-021

**実装内容**:
- [ ] CLIに `--method lazy` オプション追加
- [ ] API に lazy search エンドポイント追加
- [ ] 設定ファイルにlazy_search設定追加

**受入基準**:
- [ ] CLIからLazySearch実行可能
- [ ] APIからLazySearch実行可能

---

## Phase 5: ドキュメント・仕上げ

### TASK-LAZY-001-040: ドキュメント作成
**Priority**: P2
**Estimated**: 2h
**Files**: 
- `docs/query/lazy_search.md`
- `docs/examples_notebooks/lazy_search.ipynb`

**実装内容**:
- [ ] LazySearch使用方法ドキュメント
- [ ] Jupyter Notebookサンプル
- [ ] パラメータ説明

**受入基準**:
- [ ] 既存ドキュメント形式に準拠
- [ ] サンプルが実行可能

---

## 依存関係グラフ

```
Phase 1 (並列実行可能)
├── TASK-001 [State]
├── TASK-002 [Config]
└── TASK-003 [Prompts]
        │
        ▼
Phase 2 (Phase 1完了後、並列実行可能)
├── TASK-010 [QueryExpander] ← TASK-003
├── TASK-011 [RelevanceTester] ← TASK-001, TASK-003
├── TASK-012 [ClaimExtractor] ← TASK-001, TASK-003
├── TASK-013 [ContextBuilder] ← TASK-001
└── TASK-014 [IterativeDeepener] ← TASK-011, TASK-013
        │
        ▼
Phase 3 (Phase 2完了後)
├── TASK-020 [LazySearch] ← 全Phase 2
└── TASK-021 [__init__.py] ← TASK-020
        │
        ▼
Phase 4 (Phase 3完了後)
├── TASK-030 [Unit Tests] ← Phase 3
└── TASK-031 [Integration Tests] ← TASK-030
        │
        ▼
Phase 5 (Phase 4完了後)
├── TASK-032 [CLI/API] ← TASK-021
└── TASK-040 [Docs] ← Phase 4
```

---

## 見積もりサマリー

| Phase | タスク数 | 合計見積もり |
|-------|----------|--------------|
| Phase 1 | 3 | 5h |
| Phase 2 | 5 | 18h |
| Phase 3 | 2 | 7h |
| Phase 4 | 3 | 13h |
| Phase 5 | 1 | 2h |
| **Total** | **14** | **45h** |

---

## 実装順序推奨

1. **Day 1**: Phase 1 全タスク（基盤）
2. **Day 2-3**: Phase 2 タスク（コンポーネント）
3. **Day 4**: Phase 3 タスク（統合）
4. **Day 5**: Phase 4 タスク（テスト）
5. **Day 6**: Phase 5 タスク（仕上げ）

---

**Powered by MUSUBI** - Constitutional governance for specification-driven development.
