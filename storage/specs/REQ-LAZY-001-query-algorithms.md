# LazyGraphRAG クエリアルゴリズム要件定義書

**Document ID**: REQ-LAZY-001
**Version**: 1.0
**Status**: Draft
**Created**: 2024-12-22
**Author**: GitHub Copilot (MUSUBI SDD)

---

## 1. 概要

### 1.1 目的

本ドキュメントは、LazyGraphRAG固有のクエリアルゴリズムを実装するための要件を定義します。既存のGraphRAGクエリエンジン（Local Search、Global Search、DRIFT Search）に加え、LazyGraphRAG方式の検索機能を追加します。

### 1.2 スコープ

- サブクエリ分解（Subquery Decomposition）
- 関連性テスト（Relevance Testing）
- 反復的深化検索（Iterative Deepening Search）
- 主張抽出（Claim Extraction）
- 関連性テスト予算パラメータ（Relevance Test Budget）

### 1.3 参照文献

- Microsoft Research Blog: "LazyGraphRAG: Setting a new standard for quality and cost in RAG"
- Microsoft Discovery GraphRAG Documentation
- 既存GraphRAG実装: `graphrag/query/structured_search/`

---

## 2. 機能要件

### 2.1 サブクエリ分解 (REQ-LAZY-001-001 〜 REQ-LAZY-001-005)

#### REQ-LAZY-001-001: クエリ展開
**パターン**: Event-driven

```
WHEN ユーザーがクエリを送信した時、
LazyGraphRAG検索システム SHALL
クエリをLLMに渡し、関連するサブクエリを識別して
単一の展開クエリに変換する。
```

**受入基準**:
- [ ] 入力クエリに対して、LLMがサブクエリのリストを生成できる
- [ ] サブクエリが単一の展開クエリに結合される
- [ ] 展開クエリが元のクエリの意図を保持している

#### REQ-LAZY-001-002: サブクエリ優先順位付け
**パターン**: Ubiquitous

```
LazyGraphRAG検索システム SHALL
生成されたサブクエリを意味的関連性に基づいて
優先順位付けする。
```

**受入基準**:
- [ ] サブクエリにスコアが割り当てられる
- [ ] スコアに基づいてサブクエリがソートされる
- [ ] 関連性の低いサブクエリがフィルタリングされる

#### REQ-LAZY-001-003: クエリ拡張の設定
**パターン**: Optional features

```
WHERE クエリ拡張機能が有効な場合、
LazyGraphRAG検索システム SHALL
設定されたmax_subqueriesパラメータに基づいて
サブクエリの数を制限する。
```

**受入基準**:
- [ ] max_subqueriesパラメータが設定可能
- [ ] デフォルト値は5
- [ ] 1〜20の範囲で設定可能

---

### 2.2 クエリマッチング (REQ-LAZY-001-010 〜 REQ-LAZY-001-015)

#### REQ-LAZY-001-010: ベストファースト検索
**パターン**: Event-driven

```
WHEN 展開クエリが準備された時、
LazyGraphRAG検索システム SHALL
テキストチャンクの埋め込みを使用して
最も関連性の高いコミュニティを識別する。
```

**受入基準**:
- [ ] テキストチャンク埋め込みとクエリ埋め込みのコサイン類似度が計算される
- [ ] 上位K個のテキストチャンクが特定される
- [ ] テキストチャンクが属するコミュニティが識別される

#### REQ-LAZY-001-011: コミュニティランキング
**パターン**: Ubiquitous

```
LazyGraphRAG検索システム SHALL
各コミュニティに対して、含まれるテキストチャンクの
関連性スコアの合計に基づいてランキングを付与する。
```

**受入基準**:
- [ ] コミュニティごとの累積スコアが計算される
- [ ] コミュニティがスコア順にソートされる
- [ ] 階層的なコミュニティ構造が考慮される

#### REQ-LAZY-001-012: 幅優先探索統合
**パターン**: State-driven

```
WHILE ベストファースト検索を実行中、
LazyGraphRAG検索システム SHALL
各レベルで幅優先探索を適用して
関連コミュニティを網羅的に探索する。
```

**受入基準**:
- [ ] 各コミュニティレベルで幅優先探索が実行される
- [ ] 探索範囲がbreadth_limitパラメータで制御される
- [ ] 探索が階層的に進行する

---

### 2.3 関連性テスト (REQ-LAZY-001-020 〜 REQ-LAZY-001-030)

#### REQ-LAZY-001-020: 文レベル関連性評価
**パターン**: Event-driven

```
WHEN テキストチャンクが候補として選択された時、
LazyGraphRAG検索システム SHALL
LLMを使用してチャンク内の各文の
クエリに対する関連性を評価する。
```

**受入基準**:
- [ ] テキストチャンクが文に分割される
- [ ] 各文がLLMによって関連性評価される
- [ ] 関連性スコア（0-10スケール）が各文に割り当てられる

#### REQ-LAZY-001-021: 関連性テスト予算
**パターン**: Ubiquitous

```
LazyGraphRAG検索システム SHALL
関連性テスト予算パラメータ（Z）に基づいて
LLM呼び出し回数を制限する。
```

**受入基準**:
- [ ] Zパラメータが設定可能
- [ ] プリセット値: Z100, Z500, Z1500
- [ ] カスタム値も設定可能
- [ ] 予算超過時に検索が適切に終了する

#### REQ-LAZY-001-022: 関連性テスト効率化
**パターン**: Optional features

```
WHERE バッチ処理が有効な場合、
LazyGraphRAG検索システム SHALL
複数の文を1回のLLM呼び出しで評価する。
```

**受入基準**:
- [ ] batch_sizeパラメータが設定可能（デフォルト: 10）
- [ ] バッチ内の全ての文が同時に評価される
- [ ] 応答が正しくパースされる

#### REQ-LAZY-001-023: 関連性スコア閾値
**パターン**: State-driven

```
WHILE 関連性テストを実行中、
LazyGraphRAG検索システム SHALL
設定された閾値を超える文のみを
関連文として保持する。
```

**受入基準**:
- [ ] relevance_thresholdパラメータが設定可能（デフォルト: 5）
- [ ] 閾値未満の文がフィルタリングされる
- [ ] フィルタリング結果が記録される

---

### 2.4 反復的深化検索 (REQ-LAZY-001-040 〜 REQ-LAZY-001-050)

#### REQ-LAZY-001-040: サブコミュニティ探索
**パターン**: Event-driven

```
WHEN 現在のコミュニティで連続してゼロ関連性が検出された時、
LazyGraphRAG検索システム SHALL
そのコミュニティのサブコミュニティに再帰的に探索を拡張する。
```

**受入基準**:
- [ ] 連続ゼロ関連性のカウントが追跡される
- [ ] zero_relevance_thresholdを超えた場合に再帰が開始される（デフォルト: 3）
- [ ] サブコミュニティが適切に取得される

#### REQ-LAZY-001-041: 深化制限
**パターン**: Ubiquitous

```
LazyGraphRAG検索システム SHALL
反復深化の最大深度をmax_depthパラメータで制限する。
```

**受入基準**:
- [ ] max_depthパラメータが設定可能（デフォルト: 3）
- [ ] 最大深度到達時に探索が停止する
- [ ] 現在の深度が追跡される

#### REQ-LAZY-001-042: 早期終了条件
**パターン**: State-driven

```
WHILE 反復的深化検索を実行中、
LazyGraphRAG検索システム SHALL
十分な関連情報が収集された場合に
検索を早期終了する。
```

**受入基準**:
- [ ] 収集された関連文の数が追跡される
- [ ] sufficient_relevance_countを超えた場合に終了（デフォルト: 50）
- [ ] 早期終了が結果に記録される

#### REQ-LAZY-001-043: 探索状態管理
**パターン**: Ubiquitous

```
LazyGraphRAG検索システム SHALL
探索状態（訪問済みコミュニティ、関連文、予算残量）を
管理する専用の状態オブジェクトを維持する。
```

**受入基準**:
- [ ] LazySearchStateクラスが実装される
- [ ] 訪問済みコミュニティのセットが維持される
- [ ] 収集された関連文のリストが維持される
- [ ] 残り予算が追跡される

---

### 2.5 主張抽出 (REQ-LAZY-001-060 〜 REQ-LAZY-001-070)

#### REQ-LAZY-001-060: コンセプトサブグラフ構築
**パターン**: Event-driven

```
WHEN 関連文が収集された時、
LazyGraphRAG検索システム SHALL
関連文に含まれるエンティティと関係から
コンセプトサブグラフを構築する。
```

**受入基準**:
- [ ] 関連文からエンティティが抽出される
- [ ] エンティティ間の関係が識別される
- [ ] サブグラフがNetworkXグラフとして構築される

#### REQ-LAZY-001-061: 主張の抽出
**パターン**: Event-driven

```
WHEN コンセプトサブグラフが構築された時、
LazyGraphRAG検索システム SHALL
LLMを使用して関連チャンクから
クエリに関連する主張を抽出する。
```

**受入基準**:
- [ ] 主張抽出プロンプトが定義される
- [ ] 各関連チャンクから主張が抽出される
- [ ] 主張がクエリとの関連性でランク付けされる

#### REQ-LAZY-001-062: 主張の重複除去
**パターン**: Ubiquitous

```
LazyGraphRAG検索システム SHALL
抽出された主張の意味的重複を検出し、
重複を除去した主張リストを生成する。
```

**受入基準**:
- [ ] 主張間の意味的類似度が計算される
- [ ] 類似度閾値を超える主張が統合される
- [ ] 統合された主張の出典が保持される

---

### 2.6 応答生成 (REQ-LAZY-001-080 〜 REQ-LAZY-001-090)

#### REQ-LAZY-001-080: Reduce Phase応答生成
**パターン**: Event-driven

```
WHEN 主張リストが完成した時、
LazyGraphRAG検索システム SHALL
LLMを使用して展開クエリに対する
最終応答を生成する。
```

**受入基準**:
- [ ] 応答生成プロンプトが定義される
- [ ] 主張リストがコンテキストとして提供される
- [ ] 応答が展開クエリの全ての側面をカバーする

#### REQ-LAZY-001-081: 出典引用
**パターン**: Ubiquitous

```
LazyGraphRAG検索システム SHALL
生成された応答に使用された主張の
出典（テキストチャンクID）を含める。
```

**受入基準**:
- [ ] 各主張の出典チャンクIDが追跡される
- [ ] 応答に出典が付記される
- [ ] 出典が検証可能である

#### REQ-LAZY-001-082: ストリーミング応答
**パターン**: Optional features

```
WHERE ストリーミングモードが有効な場合、
LazyGraphRAG検索システム SHALL
応答をトークン単位でストリーミングする。
```

**受入基準**:
- [ ] AsyncGenerator形式で応答が返される
- [ ] 各トークンが即座にyieldされる
- [ ] コールバックが適切に呼び出される

---

## 3. 非機能要件

### 3.1 パフォーマンス

#### REQ-LAZY-001-NF-001: 応答時間
**パターン**: Ubiquitous

```
LazyGraphRAG検索システム SHALL
Z500予算設定で平均応答時間が30秒以内である。
```

**受入基準**:
- [ ] ベンチマークテストで平均応答時間を計測
- [ ] 30秒を超える場合は警告を出力

#### REQ-LAZY-001-NF-002: トークン効率
**パターン**: Ubiquitous

```
LazyGraphRAG検索システム SHALL
同等の回答品質においてGlobal Searchの
10%以下のトークンを使用する。
```

**受入基準**:
- [ ] トークン使用量が記録される
- [ ] Global Searchとの比較レポートが生成可能

### 3.2 拡張性

#### REQ-LAZY-001-NF-003: カスタムプロンプト
**パターン**: Optional features

```
WHERE カスタムプロンプトが提供された場合、
LazyGraphRAG検索システム SHALL
デフォルトプロンプトを置き換えて使用する。
```

**受入基準**:
- [ ] 関連性テストプロンプトがカスタマイズ可能
- [ ] 主張抽出プロンプトがカスタマイズ可能
- [ ] 応答生成プロンプトがカスタマイズ可能

### 3.3 互換性

#### REQ-LAZY-001-NF-004: 既存APIとの整合性
**パターン**: Ubiquitous

```
LazyGraphRAG検索システム SHALL
既存のBaseSearchインターフェースを継承し、
search()およびstream_search()メソッドを提供する。
```

**受入基準**:
- [ ] BaseSearchクラスを継承
- [ ] SearchResult形式で結果を返す
- [ ] 既存のQueryCallbacksと互換

---

## 4. 制約条件

### 4.1 技術的制約

- 既存のnoun graph（`build_noun_graph.py`）を使用する
- 既存のコミュニティ検出結果を活用する
- ChatModelプロトコルに準拠するLLMを使用する

### 4.2 依存関係

- `graphrag.query.structured_search.base.BaseSearch`
- `graphrag.index.operations.build_noun_graph`
- `graphrag.query.context_builder.builders`
- `graphrag.callbacks.query_callbacks.QueryCallbacks`

---

## 5. トレーサビリティマトリックス

| 要件ID | 設計 | 実装 | テスト |
|--------|------|------|--------|
| REQ-LAZY-001-001 | TBD | TBD | TBD |
| REQ-LAZY-001-002 | TBD | TBD | TBD |
| REQ-LAZY-001-003 | TBD | TBD | TBD |
| REQ-LAZY-001-010 | TBD | TBD | TBD |
| REQ-LAZY-001-011 | TBD | TBD | TBD |
| REQ-LAZY-001-012 | TBD | TBD | TBD |
| REQ-LAZY-001-020 | TBD | TBD | TBD |
| REQ-LAZY-001-021 | TBD | TBD | TBD |
| REQ-LAZY-001-022 | TBD | TBD | TBD |
| REQ-LAZY-001-023 | TBD | TBD | TBD |
| REQ-LAZY-001-040 | TBD | TBD | TBD |
| REQ-LAZY-001-041 | TBD | TBD | TBD |
| REQ-LAZY-001-042 | TBD | TBD | TBD |
| REQ-LAZY-001-043 | TBD | TBD | TBD |
| REQ-LAZY-001-060 | TBD | TBD | TBD |
| REQ-LAZY-001-061 | TBD | TBD | TBD |
| REQ-LAZY-001-062 | TBD | TBD | TBD |
| REQ-LAZY-001-080 | TBD | TBD | TBD |
| REQ-LAZY-001-081 | TBD | TBD | TBD |
| REQ-LAZY-001-082 | TBD | TBD | TBD |

---

## 6. 変更履歴

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-12-22 | GitHub Copilot | 初版作成 |

---

**Powered by MUSUBI** - Constitutional governance for specification-driven development.
