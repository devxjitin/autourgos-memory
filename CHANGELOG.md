# Changelog

## [2.1.0] - 2026-09-09

- **Fixed:** `BaseRetriever.clear()` is now an abstract method (all real subclasses already implemented it, but nothing enforced it). `add_document()` could not be made abstract the same way -- `EpisodicMemory` deliberately has no `add_document()` (it writes through its own `remember()` API) and would have become uninstantiable -- so instead `RetrievalAugmentedMemory.__init__` now validates its retriever has `add_document()` up front, turning a deferred `AttributeError` on first use into an immediate, clear `TypeError` at construction.
- **Docs:** `BaseMemory`'s docstring now documents that `get_messages()` is intentionally not part of the interface, and explains the shape split between backends (`List[Dict]` for the buffer family vs `List[MemoryMessage]` for the rest) that was previously undocumented.
- Added a `get_context()`/`format_for_llm()` deprecation-shim test (the `add_ai_message`/`add_agent_message` half was already tested; this half was not). Removed unused imports.

## [2.0.0] - 2026-09-07

- **Breaking:** Consolidated the 7 sibling memory packages (`autourgos-buffer-memory`, `autourgos-local-memory`, `autourgos-semantic-memory`, `autourgos-summary-memory`, `autourgos-token-memory`, `autourgos-vector-memory`, `autourgos-episodic-memory`) into this package as submodules (`autourgos_memory.buffer`, `.local`, `.semantic`, `.summary`, `.token`, `.vector`, `.episodic`). All their public classes are now direct, always-available imports from `autourgos_memory` (previously "soft re-exports" that only resolved if the sibling package happened to be installed).
- **Breaking:** `pip install autourgos-buffer-memory` etc. no longer needed (and those packages are being decommissioned) — everything lives in `autourgos-memory` now. `VectorMemory`/`VectorRetriever` require the new `autourgos-memory[vector]` extra (numpy); `TokenBufferedMemory`'s accurate token counting requires the new `autourgos-memory[tiktoken]` extra. `pip install autourgos-memory[all]` installs both.
- Migration: replace `from autourgos_buffer_memory import X` (and the other 6 sibling-package imports) with `from autourgos_memory import X` — the class names are unchanged.

## [1.2.0] - 2026-09-05

- **Breaking (for external `BaseMemory` subclasses):** `add_message(role, content, timestamp=None)` is now abstract on `BaseMemory`. `add_user_message`/`add_agent_message`/`add_tool_message` are concrete defaults built on it; any subclass implementing only `add_user_message`/etc. directly (without `add_message`) will fail to instantiate.
- Added: `add_system_message()` default on `BaseMemory` (built on `add_message`).
- Added: `format_conversation_banner(messages, *, include_timestamps=True)` -- shared "Previous Conversation Context" rendering, extracted from the identical logic duplicated across every concrete backend's `format_for_llm()`.
- Added: `ROLE_TO_OPENAI` mapping (`{"user": "user", "agent": "assistant", "system": "system", "tool": "tool"}`), extracted from `autourgos-buffer-memory`'s duplicated inline dict.
- Added: `RetrievalAugmentedMemory` base class -- the shared dual-store (short-term buffer + retriever) shape behind `autourgos-semantic-memory`'s `KeywordMemory` and `autourgos-vector-memory`'s `VectorMemory`.

## [1.1.0] - 2026-09-04

- Internal: `__version__` resolution moved to `autourgos_core.package_version()` (new `autourgos-core>=0.3.0` dependency). No functional change.

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
