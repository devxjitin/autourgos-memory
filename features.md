# autourgos-memory — Features

The foundation package of the Autourgos memory family. It defines the abstract interfaces (`BaseMemory`, `BaseRetriever`, `MemoryMessage`, `Document`) that every concrete memory backend in the family implements, plus a "soft re-export" mechanism so consumers can `import` any installed concrete backend directly from `autourgos_memory` without knowing which sibling package it actually lives in. Zero required dependencies.

## Full Feature List

- **Abstract interfaces**:
  - `BaseMemory` — short-term conversational memory contract (`add_user_message`, `add_agent_message`, `add_tool_message`, `format_for_llm`, `clear`); only `add_user_message`, `add_tool_message`, and `clear` are true `@abstractmethod`s
  - `BaseRetriever` — relevance-scored recall contract (`retrieve(query, top_k) -> list[Document]`)
  - `MemoryMessage` — typed message dataclass with `role` (`user`/`agent`/`system`/`tool`), `content`, `timestamp`, and `.to_dict()`
  - `Document` — retrieval result shape (`content`, `score`, `source`)
- **Soft re-exports** — `RuntimeShortTermMemory`, `ConversationBufferMemory`, `LocalShortTermMemory`, `SQLiteMemory`, `KeywordRetriever`, `KeywordMemory`, `SimpleSemanticRetriever`, `HierarchicalSemanticMemory`, `SummaryBufferedMemory`, `TokenBufferedMemory`, `VectorMemory`, `VectorRetriever`, `Episode`, `EpisodicMemory` all resolve directly from `autourgos_memory` if their own concrete package is installed — install only what you need
- **Backward-compatibility shim** — `add_agent_message`/`format_for_llm` fall back to calling an older method name (`add_ai_message`/`get_context`) if a subclass implements that instead, with a `DeprecationWarning`, so pre-rename backends keep working unchanged
- Zero required dependencies — this package alone installs no concrete backend, only the contracts

## Competitor Comparison

This package's closest comparison isn't a single competing product but the "base memory interface/protocol" layer that other agent frameworks bundle inside themselves rather than ship as a separate installable package.

| Capability | **autourgos-memory** | [LangChain `BaseMemory`/`BaseChatMessageHistory`](https://python.langchain.com/) | [LlamaIndex memory/storage abstractions](https://docs.llamaindex.ai/) | [LangGraph `BaseCheckpointSaver`/`BaseStore`](https://langchain-ai.github.io/langgraph/) | Bespoke internal `Protocol`/ABC per project |
|---|---|---|---|---|---|
| Distributed as its own installable package | Yes — `pip install autourgos-memory` alone, no concrete backend forced | No — bundled inside `langchain-core`, always pulls in the framework | No — bundled inside `llama-index-core` | No — bundled inside `langgraph`/`langgraph-checkpoint` | N/A |
| Zero required dependencies | Yes | No — pulls in `langchain-core`'s own dependency tree | No — pulls in `llama-index-core`'s dependency tree | No — pulls in LangGraph's core | Yes (if truly bespoke) |
| Soft/optional resolution of concrete implementations from the base package | Yes — a documented design feature (`from autourgos_memory import X` resolves only if X's package is installed) | No — concrete memory classes are imported from their own submodules directly | No — concrete stores imported from their own submodules directly | No — concrete savers (`SqliteSaver`, `PostgresSaver`) imported directly | Depends on project |
| Backward-compatibility shim for renamed methods | Yes, explicit deprecation-warning fallback | Handled via LangChain's own broader deprecation tooling across the framework | Handled ad hoc per release | Handled ad hoc per release | Rare, unless deliberately designed |
| Framework lock-in | None — interfaces only, any agent runtime can implement/consume them | High — tied to LangChain's `Runnable`/chain ecosystem | High — tied to LlamaIndex's indexing/query pipeline | High — tied to LangGraph's graph/state model | None, but also no ecosystem of matching backends |
| Ecosystem of ready-made concrete backends | Yes — 7 sibling Autourgos packages (buffer, local, semantic, summary, token, vector, episodic) | Yes — very large, many third-party integrations | Yes — very large, many third-party integrations | Growing — checkpoint/store backends for major DBs | None by default |
| Pricing | Free, open source | Free, open source | Free, open source | Free, open source | N/A |

### How to read this

- **The real differentiator is decoupling**: LangChain, LlamaIndex, and LangGraph all define a memory/storage interface, but it only exists as part of installing their much larger framework — you cannot depend on "just the interface." autourgos-memory is deliberately split out so an agent runtime (or a completely unrelated project) can code against `BaseMemory`/`BaseRetriever` without a heavy framework dependency.
- **vs. rolling a bespoke Protocol per project**: the trade-off is the usual "shared interface vs. reinventing it" one — autourgos-memory buys a tested contract plus seven ready-made concrete implementations behind it (buffer/local/semantic/summary/token/vector/episodic), at the cost of adopting its conventions (`MemoryMessage` roles, `Document` shape).
- **Not a retrieval or storage engine itself**: this package makes no claims about vector search, embeddings, or SQL — those live in the concrete sibling packages. Anyone evaluating it against Mem0/Zep/Letta is really evaluating the *whole Autourgos memory family* those siblings form, not this base package in isolation.
- **Soft re-export is a genuine, somewhat unusual design choice**: most Python ecosystems either force a hard dependency or require importing from the specific sub-package; conditionally resolving names from the base package only when siblings happen to be installed is not something LangChain/LlamaIndex/LangGraph do the same way.

Sources:
- [LangGraph Memory vs Mem0: Which Should You Use in 2026?](https://atlan.com/know/ai-agent/ai-agent-memory/langgraph-memory-vs-mem0/)
- [Best AI Agent Memory Frameworks in 2026: Compared and Ranked](https://atlan.com/know/best-ai-agent-memory-frameworks-2026/)
- [LangGraph vs LangChain: Which to Use for Production AI Agents in 2026](https://www.spheron.network/blog/langgraph-vs-langchain/)
- [LangChain Memory Component Deep Dive: Chain Components and Runnable Study](https://dev.to/jamesli/langchain-memory-component-deep-dive-chain-components-and-runnable-study-359p)
