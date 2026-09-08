# autourgos-memory — Features

The single, unified package of the Autourgos memory family. It defines the abstract interfaces (`BaseMemory`, `BaseRetriever`, `MemoryMessage`, `Document`) plus every concrete memory backend that used to live in seven separate sibling packages (buffer, local, semantic, summary, token, vector, episodic) — now submodules of this one package, importable directly with no soft-resolution step.

## Full Feature List

- **Abstract interfaces**:
  - `BaseMemory` — short-term conversational memory contract (`add_user_message`, `add_agent_message`, `add_tool_message`, `format_for_llm`, `clear`); only `add_message`, `format_for_llm`, and `clear` are true `@abstractmethod`s
  - `BaseRetriever` — relevance-scored recall contract (`retrieve(query, top_k) -> list[Document]`)
  - `MemoryMessage` — typed message dataclass with `role` (`user`/`agent`/`system`/`tool`), `content`, `timestamp`, and `.to_dict()`
  - `Document` — retrieval result shape (`content`, `score`, `source`)
- **Every concrete backend bundled** — `RuntimeShortTermMemory`, `ConversationBufferMemory`, `ExpiringBufferMemory` (`autourgos_memory.buffer`); `LocalShortTermMemory`, `SQLiteMemory` (`.local`); `KeywordRetriever`, `KeywordMemory` (`.semantic`); `SummaryBufferedMemory` (`.summary`); `TokenBufferedMemory` (`.token`); `VectorMemory`, `VectorRetriever` (`.vector`, needs the `[vector]` extra); `Episode`, `EpisodicMemory` (`.episodic`) — all also exported from top-level `autourgos_memory`
- **Backward-compatibility shim** — `add_agent_message`/`format_for_llm` fall back to calling an older method name (`add_ai_message`/`get_context`) if a subclass implements that instead, with a `DeprecationWarning`, so pre-rename backends keep working unchanged
- Only two optional dependencies, both opt-in via extras: `numpy` (`[vector]`, for `VectorMemory`/`VectorRetriever`) and `tiktoken` (`[tiktoken]`, for accurate `TokenBufferedMemory` counting — falls back to a heuristic estimator without it)

## Competitor Comparison

This package's closest comparison isn't a single competing product but the "base memory interface/protocol" layer that other agent frameworks bundle inside themselves rather than ship as a separate installable package.

| Capability | **autourgos-memory** | [LangChain `BaseMemory`/`BaseChatMessageHistory`](https://python.langchain.com/) | [LlamaIndex memory/storage abstractions](https://docs.llamaindex.ai/) | [LangGraph `BaseCheckpointSaver`/`BaseStore`](https://langchain-ai.github.io/langgraph/) | Bespoke internal `Protocol`/ABC per project |
|---|---|---|---|---|---|
| Distributed as its own installable package | Yes — `pip install autourgos-memory` installs the interfaces and all 7 backends together | No — bundled inside `langchain-core`, always pulls in the framework | No — bundled inside `llama-index-core` | No — bundled inside `langgraph`/`langgraph-checkpoint` | N/A |
| Minimal required dependencies | Yes — only `autourgos-core`; `numpy`/`tiktoken` are opt-in extras (`[vector]`/`[tiktoken]`) | No — pulls in `langchain-core`'s own dependency tree | No — pulls in `llama-index-core`'s dependency tree | No — pulls in LangGraph's core | Yes (if truly bespoke) |
| Backward-compatibility shim for renamed methods | Yes, explicit deprecation-warning fallback | Handled via LangChain's own broader deprecation tooling across the framework | Handled ad hoc per release | Handled ad hoc per release | Rare, unless deliberately designed |
| Framework lock-in | None — interfaces only, any agent runtime can implement/consume them | High — tied to LangChain's `Runnable`/chain ecosystem | High — tied to LlamaIndex's indexing/query pipeline | High — tied to LangGraph's graph/state model | None, but also no ecosystem of matching backends |
| Ecosystem of ready-made concrete backends | Yes — 7 backends bundled in this one package (buffer, local, semantic, summary, token, vector, episodic) | Yes — very large, many third-party integrations | Yes — very large, many third-party integrations | Growing — checkpoint/store backends for major DBs | None by default |
| Pricing | Free, open source | Free, open source | Free, open source | Free, open source | N/A |

### How to read this

- **The real differentiator is decoupling**: LangChain, LlamaIndex, and LangGraph all define a memory/storage interface, but it only exists as part of installing their much larger framework — you cannot depend on "just the interface." autourgos-memory is deliberately split out so an agent runtime (or a completely unrelated project) can code against `BaseMemory`/`BaseRetriever` without a heavy framework dependency.
- **vs. rolling a bespoke Protocol per project**: the trade-off is the usual "shared interface vs. reinventing it" one — autourgos-memory buys a tested contract plus seven ready-made concrete implementations behind it (buffer/local/semantic/summary/token/vector/episodic), at the cost of adopting its conventions (`MemoryMessage` roles, `Document` shape).
- **Not a retrieval or storage engine itself**: this package makes no claims about vector search, embeddings, or SQL beyond what its own bundled backends implement. Anyone evaluating it against Mem0/Zep/Letta is evaluating the *whole* package, since all 7 backends now ship together rather than as separate installs.
- **Consolidated, not soft-resolved**: earlier versions resolved concrete backends only if a matching sibling package happened to be installed; as of 2.0.0 every backend (bar the numpy/tiktoken-dependent ones) is a hard, always-available import from this single package — one install, no partial states.

Sources:
- [LangGraph Memory vs Mem0: Which Should You Use in 2026?](https://atlan.com/know/ai-agent/ai-agent-memory/langgraph-memory-vs-mem0/)
- [Best AI Agent Memory Frameworks in 2026: Compared and Ranked](https://atlan.com/know/best-ai-agent-memory-frameworks-2026/)
- [LangGraph vs LangChain: Which to Use for Production AI Agents in 2026](https://www.spheron.network/blog/langgraph-vs-langchain/)
- [LangChain Memory Component Deep Dive: Chain Components and Runnable Study](https://dev.to/jamesli/langchain-memory-component-deep-dive-chain-components-and-runnable-study-359p)
