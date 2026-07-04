# Simplifications vs Production

| Area | This assignment | Production |
|------|----------------|------------|
| DB access control | Read-only `api_reader` role for the API; indexer still connects as the full DB owner | Dedicated least-privilege write role for the indexer too, not just the API |
| API security | No auth | JWT / API key middleware + rate limiting |
| Secrets | Plain env vars | Secrets manager (Vault, AWS Secrets Manager) |
| Schema changes | `init.sql` runs once | Alembic versioned migrations |
| Connection pooling | Shared async engine per process | PgBouncer for pooling across multiple API processes/replicas |
| Vector index | No ANN index — exact KNN via sequential scan (see `db/init.sql`); deliberate at this corpus scale (an ivfflat index over few vectors leaves most lists empty, returning 0 rows for novel queries) | HNSW index once the corpus is large enough to need it |
| Embedding model | `all-MiniLM-L6-v2` (local, generic) | `voyage-finance-2` (Voyage AI — Anthropic's embedding partner, domain-specific for financial docs) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-12-v2` (local, generic, MS MARCO-trained) — scores broad/summary questions near zero even against relevant content, only mitigated by picking cosine similarity instead for that question type | Domain-tuned reranker (e.g. a cross-encoder fine-tuned on financial QA pairs) that doesn't need a query-type-dependent fallback |
| Similarity threshold | Fixed absolute cutoff (`SIMILARITY_THRESHOLD = 0.30`) for `broad`-style queries — correctly-classified topic-existence questions ("is there anything about X") can still land just under it depending on exact phrasing: two near-identical facts about the same sentence scored 0.306 (passed) and 0.225 (failed) for differently-worded questions, confirmed via direct embedding comparison. Not a classification bug (verified both Sonnet 5 and Opus 4.8 correctly classify these as `broad`) — the fixed number is just a blunt instrument at this scale | Calibrate the threshold on a larger labelled eval set, or use a relative/percentile cutoff (e.g. top-score-relative-to-corpus-median) instead of one fixed absolute number |
| Embedding versioning | None | Store model version per embedding; detect mismatches and re-index |
| LLM retries | SDK default only — `ChatAnthropic(max_retries=2)`, the Anthropic client's built-in exponential backoff on 429/5xx/timeouts. No custom backoff policy, no retry-count visibility/metrics, no circuit breaker | Configurable backoff policy, retry metrics/alerting, circuit breaker on sustained provider failure |
| Eval thresholds | Hardcoded constants | Calibrated on a labelled eval set, stored in config |
| Table numeric parsing | Normalised at index time (non-lossy — see note below) | Store cells verbatim; normalise at query time via a shared parser so stored data stays the audit source |
| Observability | None | Structured logs, OpenTelemetry tracing, Prometheus metrics |
| Scaling | Single instance | Horizontal API replicas, Postgres read replicas |

---

## Note: table numeric normalization

Postgres and pandas can't aggregate over the strings a financial PDF actually
contains — `'2,450'::numeric`, `'(500)'::numeric`, and `'$1,420'::numeric` all
error — so the analytical examples ("sum of Q3 headcount = 8,200", "highest Q4
headcount = R&D 2,620") force numeric normalization *somewhere*. It can't be
skipped; it can only live at index time or at query time.

**This assignment — normalize once at index time, non-destructively** (`indexer/pdf_parser.py::_df_to_rows`, `_coerce_number`):

| PDF cell | Stored | Why |
|----------|--------|-----|
| `2,450` | `2450` | thousands separator stripped |
| `$1,420` | `1420` | currency symbol stripped |
| `(500)` | `-500` | accounting negative — **must not** become `NaN→None`, which would drop a real figure and bias every `SUM`/`AVG` upward |
| `12%` | `"12%"` | kept as string — coercing to bare `12` would let a later `SUM`/`AVG` mis-add a rate as a quantity |
| `N/A` (in a numeric column) | `"N/A"` | non-numeric cell preserved verbatim, never nulled |
| `` (empty) | `None` | only genuinely blank cells are null |

Invariant: **no non-empty source cell silently becomes `None`.**

**Production — store cells verbatim, normalize at query time.** Keep `row_data`
exactly as it appears in the PDF (`"2,450"`, `"(500)"`, `"12%"`) and convert only
when computing, via one deterministic, tested helper: a `to_num(text)→numeric`
SQL function for the analytical path plus the equivalent Python parser for the
hybrid path (both returning `NULL`/`None` on anything unparseable, e.g. suffixed
magnitudes like `"1.42B"`, rather than guessing a multiplier). This makes the
stored data the verbatim audit source and confines normalization to a single
place — at the cost of a schema/`init.sql` change and keeping the two parsers in
lockstep, so it's deferred here.
