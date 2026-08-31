# Changelog

## [1.0.6] - 2026-08-31

- Added: soft re-export of `VectorMemory`/`VectorRetriever` from the new `autourgos-vector-memory` package (local, provider-agnostic embedding recall -- caller supplies the embedding function), matching the existing soft-re-export convention for every other concrete memory backend.

## [1.0.5] - 2026-08-31

- **Removed:** the soft re-export of `LongTermMemory` from `autourgos-longterm-memory`, added in 1.0.3. That package and the rest of the "v3" package family it belonged to have been removed from the workspace; `autourgos-memory` returns to re-exporting only `autourgos-buffer-memory`/`autourgos-local-memory`/`autourgos-semantic-memory`/`autourgos-summary-memory`/`autourgos-token-memory`.

## [1.0.3] - 2026-08-31

- Added: soft re-export of `LongTermMemory` from the new `autourgos-longterm-memory` package (SQLite-backed, keyword-scored, cross-run `BaseRetriever`), matching the existing soft-re-export convention for every other concrete memory backend.

## [1.0.2] - 2026-07-27

- Added: module logger. Docs: Quick Start now notes autourgos-buffer-memory is required for the RuntimeShortTermMemory re-export, and fixes the undefined my_llm placeholder.

## [1.0.1] - 2026-06-17

- Update Documentation
