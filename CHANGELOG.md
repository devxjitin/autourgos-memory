# Changelog

## [1.0.9] - 2026-09-03

- Added `features.md` documenting the module's feature set and a competitor comparison. No code changes.


## [1.0.8] - 2026-09-01

- Metadata: added `maintainers` (Sonia, Vishwanil Suman) to `pyproject.toml`,
  and linked the README's existing Sonia contributor badge to her GitHub
  profile (https://github.com/dahiyasonia). No code changes.

## [1.0.7] - 2026-08-31

- Added: soft re-export of `Episode`/`EpisodicMemory` from the new `autourgos-episodic-memory` package (structured task/outcome log), matching the existing soft-re-export convention for every other concrete memory backend.

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
