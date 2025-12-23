# Project Structure

**Project**: MS-GraphRAG+LazyGraphRAG
**Last Updated**: 2025-12-24
**Version**: 2.0
**Status**: ✅ 実装完了

---

## Architecture Pattern

**Primary Pattern**: Existing Codebase Extension (Brownfield)

> 既存のMicrosoft GraphRAGコードベースに、LazyGraphRAGクエリアルゴリズムを追加する
> ブラウンフィールド開発アプローチ。既存のパターンとインターフェースに準拠する。

---

## GraphRAG Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI / API Layer                          │
│                   (graphrag/cli/, graphrag/api/)                │
├─────────────────────────────────────────────────────────────────┤
│                        Query Engine                             │
│              (graphrag/query/structured_search/)                │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│   │  Local   │ │  Global  │ │  DRIFT   │ │  Lazy    │ ✅ DONE │
│   │  Search  │ │  Search  │ │  Search  │ │  Search  │         │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘         │
├─────────────────────────────────────────────────────────────────┤
│                     Context Builders                            │
│             (graphrag/query/context_builder/)                   │
├─────────────────────────────────────────────────────────────────┤
│                      Index Pipeline                             │
│                   (graphrag/index/)                             │
│   ┌──────────────────┐ ┌──────────────────┐                    │
│   │  Entity/Relation │ │  NounGraph       │ ← LazyGraphRAG     │
│   │  Extraction      │ │  (build_noun_    │                    │
│   │  (LLM-based)     │ │   graph/)        │                    │
│   └──────────────────┘ └──────────────────┘                    │
├─────────────────────────────────────────────────────────────────┤
│                     Data Model Layer                            │
│                  (graphrag/data_model/)                         │
├─────────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                          │
│  (storage/, vector_stores/, cache/, language_model/)            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Query Search Implementations

### Existing Search Types

| Search Type | Location | Description |
|-------------|----------|-------------|
| Local Search | `local_search/` | Entity-focused, knowledge graph traversal |
| Global Search | `global_search/` | Community-based map-reduce |
| DRIFT Search | `drift_search/` | Dynamic reasoning with iterative refinement |

### New: LazySearch (To Be Implemented)

| Component | File | Description |
|-----------|------|-------------|
| `LazySearch` | `search.py` | Main orchestrator |
| `LazySearchState` | `state.py` | State management |
| `LazyContextBuilder` | `context.py` | Context building |
| `QueryExpander` | `query_expander.py` | Subquery decomposition |
| `RelevanceTester` | `relevance_tester.py` | Sentence-level relevance |
| `ClaimExtractor` | `claim_extractor.py` | Claim extraction |
| `IterativeDeepener` | `iterative_deepener.py` | Recursive exploration |

---

## Directory Organization

### GraphRAG Root Structure

```
MS-GraphRAG+LazyGraphRAG/
├── graphrag/                   # Main package
│   ├── api/                    # Public API
│   ├── cli/                    # Command-line interface
│   ├── config/                 # Configuration models
│   ├── data_model/             # Data structures
│   ├── index/                  # Indexing pipeline
│   │   └── operations/
│   │       └── build_noun_graph/  # LazyGraphRAG NLP extraction
│   ├── query/                  # Query engine
│   │   ├── context_builder/    # Context builders
│   │   └── structured_search/  # Search implementations
│   │       ├── base.py         # BaseSearch class
│   │       ├── local_search/   # Local search
│   │       ├── global_search/  # Global search
│   │       ├── drift_search/   # DRIFT search
│   │       └── lazy_search/    # ← NEW: LazyGraphRAG search
│   ├── prompts/                # LLM prompts
│   └── ...
├── tests/                      # Test suites
├── docs/                       # Documentation
├── storage/                    # SDD artifacts
│   ├── specs/                  # Requirements & specifications
│   ├── changes/                # Delta specifications
│   └── features/               # Feature tracking
├── steering/                   # Project memory
│   ├── structure.ja.md         # This file
│   ├── tech.ja.md              # Technology stack
│   ├── product.ja.md           # Product context
│   ├── project.yml             # Project configuration
│   └── rules/                  # Constitutional governance
└── templates/                  # Document templates
```

---

## Library-First Pattern (Article I)

All features begin as independent libraries in `lib/`.

### Library Structure

Each library follows this structure:

```
lib/{{feature}}/
├── src/
│   ├── index.ts          # Public API exports
│   ├── service.ts        # Business logic
│   ├── repository.ts     # Data access
│   ├── types.ts          # TypeScript types
│   ├── errors.ts         # Custom errors
│   └── validators.ts     # Input validation
├── tests/
│   ├── service.test.ts   # Unit tests
│   ├── repository.test.ts # Integration tests (real DB)
│   └── integration.test.ts # E2E tests
├── cli.ts                # CLI interface (Article II)
├── package.json          # Library metadata
├── tsconfig.json         # TypeScript config
└── README.md             # Library documentation
```

### Library Guidelines

- **Independence**: Libraries MUST NOT depend on application code
- **Public API**: All exports via `src/index.ts`
- **Testing**: Independent test suite
- **CLI**: All libraries expose CLI interface (Article II)

---

## LazySearch Module Structure (New)

### Module Organization

```
graphrag/query/structured_search/lazy_search/
├── __init__.py               # Public exports
├── search.py                 # LazySearch class (main orchestrator)
├── state.py                  # LazySearchState, RelevantSentence, Claim
├── context.py                # LazyContextBuilder
├── query_expander.py         # QueryExpander (subquery decomposition)
├── relevance_tester.py       # RelevanceTester (sentence-level)
├── claim_extractor.py        # ClaimExtractor
└── iterative_deepener.py     # IterativeDeepener (recursive exploration)
```

### Module Guidelines

- **BaseSearch継承**: `LazySearch`は`BaseSearch[LazyContextBuilder]`を継承
- **状態管理**: `LazySearchState`で探索状態を一元管理
- **単一責任**: 各クラスは単一の責任を持つ
- **テスト可能**: 各コンポーネントは個別にテスト可能

---

## Test Organization

### Test Structure (GraphRAG)

```
tests/
├── unit/                     # Unit tests
│   ├── query/
│   │   └── structured_search/
│   │       ├── local_search/
│   │       ├── global_search/
│   │       ├── drift_search/
│   │       └── lazy_search/    # ← NEW
│   │           ├── test_search.py
│   │           ├── test_state.py
│   │           ├── test_query_expander.py
│   │           ├── test_relevance_tester.py
│   │           ├── test_claim_extractor.py
│   │           └── test_iterative_deepener.py
│   └── index/
│       └── operations/
│           └── build_noun_graph/
├── integration/              # Integration tests
│   └── query/
│       └── lazy_search/
│           └── test_lazy_search_e2e.py
└── fixtures/                 # Test data
    └── lazy_search/
        ├── sample_noun_graph.parquet
        └── sample_communities.parquet
```

### Testing Guidelines (Article III & IX)

- **Test-First**: テストを実装前に作成（Red-Green-Blue）
- **Integration-First**: 実サービスを使用した統合テスト優先
- **Coverage**: 80%以上のカバレッジ必須
- **Mock制限**: 外部サービス不可時のみモック使用可

---

## Configuration Files

### LazySearch Configuration

```
graphrag/config/models/
├── lazy_search_config.py     # ← NEW: LazySearchConfig dataclass
└── ...

graphrag/prompts/query/
├── lazy_query_expansion_prompt.py      # ← NEW
├── lazy_relevance_test_prompt.py       # ← NEW
├── lazy_claim_extraction_prompt.py     # ← NEW
└── lazy_response_generation_prompt.py  # ← NEW
```

---

## Specifications Location

### SDD Artifacts

```
storage/
├── specs/                    # Requirements & Specifications
│   ├── REQ-LAZY-001-query-algorithms.md  # EARS Requirements
│   └── SPEC-LAZY-001-query-algorithms.md # Functional Spec
├── changes/                  # Delta specifications
└── features/                 # Feature tracking
```

---

## Application Structure (unified-search-app)

### Application Organization

```
app/
├── (auth)/               # Route groups (Next.js App Router)
│   ├── login/
│   │   └── page.tsx
│   └── register/
│       └── page.tsx
├── dashboard/
│   └── page.tsx
├── api/                  # API routes
│   ├── auth/
│   │   └── route.ts
│   └── users/
│       └── route.ts
├── layout.tsx            # Root layout
└── page.tsx              # Home page
```

### Application Guidelines

- **Library Usage**: Applications import from `lib/` modules
- **Thin Controllers**: API routes delegate to library services
- **No Business Logic**: Business logic belongs in libraries

---

## Component Organization

### UI Components

```
components/
├── ui/                   # Base UI components (shadcn/ui)
│   ├── button.tsx
│   ├── input.tsx
│   └── card.tsx
├── auth/                 # Feature-specific components
│   ├── LoginForm.tsx
│   └── RegisterForm.tsx
├── dashboard/
│   └── StatsCard.tsx
└── shared/               # Shared components
    ├── Header.tsx
    └── Footer.tsx
```

### Component Guidelines

- **Composition**: Prefer composition over props drilling
- **Types**: All props typed with TypeScript
- **Tests**: Component tests with React Testing Library

---

## Database Organization

### Schema Organization

```
prisma/
├── schema.prisma         # Prisma schema
├── migrations/           # Database migrations
│   ├── 001_create_users_table/
│   │   └── migration.sql
│   └── 002_create_sessions_table/
│       └── migration.sql
└── seed.ts               # Database seed data
```

### Database Guidelines

- **Migrations**: All schema changes via migrations
- **Naming**: snake_case for tables and columns
- **Indexes**: Index foreign keys and frequently queried columns

---

## Test Organization

### Test Structure

```
tests/
├── unit/                 # Unit tests (per library)
│   └── auth/
│       └── service.test.ts
├── integration/          # Integration tests (real services)
│   └── auth/
│       └── login.test.ts
├── e2e/                  # End-to-end tests
│   └── auth/
│       └── user-flow.test.ts
└── fixtures/             # Test data and fixtures
    └── users.ts
```

### Test Guidelines

- **Test-First**: Tests written BEFORE implementation (Article III)
- **Real Services**: Integration tests use real DB/cache (Article IX)
- **Coverage**: Minimum 80% coverage
- **Naming**: `*.test.ts` for unit, `*.integration.test.ts` for integration

---

## Documentation Organization

### Documentation Structure

```
docs/
├── architecture/         # Architecture documentation
│   ├── c4-diagrams/
│   └── adr/              # Architecture Decision Records
├── api/                  # API documentation
│   ├── openapi.yaml
│   └── graphql.schema
├── guides/               # Developer guides
│   ├── getting-started.md
│   └── contributing.md
└── runbooks/             # Operational runbooks
    ├── deployment.md
    └── troubleshooting.md
```

---

## SDD Artifacts Organization

### Storage Directory

```
storage/
├── specs/                # Specifications
│   ├── auth-requirements.md
│   ├── auth-design.md
│   ├── auth-tasks.md
│   └── payment-requirements.md
├── changes/              # Delta specifications (brownfield)
│   ├── add-2fa.md
│   └── upgrade-jwt.md
├── features/             # Feature tracking
│   ├── auth.json
│   └── payment.json
└── validation/           # Validation reports
    ├── auth-validation-report.md
    └── payment-validation-report.md
```

---

## Naming Conventions

### File Naming

- **TypeScript**: `PascalCase.tsx` for components, `camelCase.ts` for utilities
- **React Components**: `PascalCase.tsx` (e.g., `LoginForm.tsx`)
- **Utilities**: `camelCase.ts` (e.g., `formatDate.ts`)
- **Tests**: `*.test.ts` or `*.spec.ts`
- **Constants**: `SCREAMING_SNAKE_CASE.ts` (e.g., `API_ENDPOINTS.ts`)

### Directory Naming

- **Features**: `kebab-case` (e.g., `user-management/`)
- **Components**: `kebab-case` or `PascalCase` (consistent within project)

### Variable Naming

- **Variables**: `camelCase`
- **Constants**: `SCREAMING_SNAKE_CASE`
- **Types/Interfaces**: `PascalCase`
- **Enums**: `PascalCase`

---

## Integration Patterns

### Library → Application Integration

```typescript
// ✅ CORRECT: Application imports from library
import { AuthService } from '@/lib/auth';

const authService = new AuthService(repository);
const result = await authService.login(credentials);
```

```typescript
// ❌ WRONG: Library imports from application
// Libraries must NOT depend on application code
import { AuthContext } from '@/app/contexts/auth'; // Violation!
```

### Service → Repository Pattern

```typescript
// Service layer (business logic)
export class AuthService {
  constructor(private repository: UserRepository) {}

  async login(credentials: LoginRequest): Promise<LoginResponse> {
    // Business logic here
    const user = await this.repository.findByEmail(credentials.email);
    // ...
  }
}

// Repository layer (data access)
export class UserRepository {
  constructor(private prisma: PrismaClient) {}

  async findByEmail(email: string): Promise<User | null> {
    return this.prisma.user.findUnique({ where: { email } });
  }
}
```

---

## Deployment Structure

### Deployment Units

**Projects** (independently deployable):

1. MS-GraphRAG+LazyGraphRAG - Main application

> ⚠️ **Simplicity Gate (Article VII)**: Maximum 3 projects initially.
> If adding more projects, document justification in Phase -1 Gate approval.

### Environment Structure

```
environments/
├── development/
│   └── .env.development
├── staging/
│   └── .env.staging
└── production/
    └── .env.production
```

---

## Multi-Language Support

### Language Policy

- **Primary Language**: English
- **Documentation**: English first (`.md`), then Japanese (`.ja.md`)
- **Code Comments**: English
- **UI Strings**: i18n framework

### i18n Organization

```
locales/
├── en/
│   ├── common.json
│   └── auth.json
└── ja/
    ├── common.json
    └── auth.json
```

---

## Version Control

### Branch Organization

- `main` - Production branch
- `develop` - Development branch
- `feature/*` - Feature branches
- `hotfix/*` - Hotfix branches
- `release/*` - Release branches

### Commit Message Convention

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Example**:

```
feat(auth): implement user login (REQ-AUTH-001)

Add login functionality with email and password authentication.
Session created with 24-hour expiry.

Closes REQ-AUTH-001
```

---

## Constitutional Compliance

This structure enforces:

- **Article I**: Library-first pattern in `lib/`
- **Article II**: CLI interfaces per library
- **Article III**: Test structure supports Test-First
- **Article VI**: Steering files maintain project memory

---

## Changelog

### Version 1.1 (Planned)

- [Future changes]

---

**Last Updated**: 2025-12-22
**Maintained By**: {{MAINTAINER}}
