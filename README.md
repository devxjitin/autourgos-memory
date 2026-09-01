# autourgos-memory

[![Framework: Autourgos](https://img.shields.io/badge/Framework-Autourgos-orange.svg)](https://github.com/devxjitin)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://pypi.org/project/autourgos-memory/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/devxjitin/autourgos-memory/blob/main/LICENSE)
[![Author](https://img.shields.io/badge/Author-Jitin%20Kumar%20Sengar-blue.svg)](https://github.com/devxjitin)
[![Contributor](https://img.shields.io/badge/Contributor-Sonia-blueviolet.svg)]()
[![Contributor](https://img.shields.io/badge/Contributor-Vishwanil%20Suman-blueviolet.svg)]()

Base memory interfaces for [Autourgos](https://github.com/devxjitin) agents — the **foundation package**. It
defines the abstract interfaces (`BaseMemory`, `BaseRetriever`, `MemoryMessage`, `Document`) that every
concrete memory implementation uses.

```python
from autourgos_memory import RuntimeShortTermMemory  # requires autourgos-buffer-memory installed
from autourgos_agent import Agent
from autourgos_openaichat import OpenAIChatModel

my_llm = OpenAIChatModel(model="gpt-4o-mini")
memory = RuntimeShortTermMemory(max_messages=20)
agent  = Agent(llm=my_llm, memory=memory)
```

---

## Features

- **Abstract interfaces**: `BaseMemory` (short-term conversational), `BaseRetriever` (relevance-scored
  recall), `MemoryMessage`, `Document`
- **Soft re-exports**: concrete backends resolve from `autourgos_memory` directly if their own package is
  installed — `RuntimeShortTermMemory`, `ConversationBufferMemory`, `LocalShortTermMemory`, `SQLiteMemory`,
  `KeywordRetriever`, `KeywordMemory`, `SimpleSemanticRetriever`, `HierarchicalSemanticMemory`,
  `SummaryBufferedMemory`, `TokenBufferedMemory`, `VectorMemory`, `VectorRetriever`, `Episode`,
  `EpisodicMemory`
- Zero required dependencies — install only the concrete backends you actually need

---

## Table of Contents

- [Install](#install)
- [Memory Types at a Glance](#memory-types-at-a-glance)
- [Quick Start](#quick-start)
- [Base Interfaces](#base-interfaces)
- [License](#license)

---

## Install

```bash
# Base interfaces only
pip install autourgos-memory

# Or install concrete implementations individually
pip install autourgos-buffer-memory      # in-memory ring buffer
pip install autourgos-local-memory       # JSON file + SQLite
pip install autourgos-semantic-memory    # TF-IDF keyword retrieval
pip install autourgos-summary-memory     # LLM-compressed rolling summary
pip install autourgos-token-memory       # token-bounded buffer
pip install autourgos-vector-memory      # local, provider-agnostic embedding recall
pip install autourgos-episodic-memory    # structured task/outcome log
```

---

## Memory Types at a Glance

| Package | Class | Best for |
|---|---|---|
| `autourgos-buffer-memory` | `RuntimeShortTermMemory` | Fast in-memory buffer, message-count bounded |
| `autourgos-buffer-memory` | `ConversationBufferMemory` | Unbounded in-memory buffer |
| `autourgos-local-memory` | `LocalShortTermMemory` | Disk persistence via JSON file |
| `autourgos-local-memory` | `SQLiteMemory` | Disk persistence via SQLite, concurrent-safe |
| `autourgos-semantic-memory` | `KeywordMemory` | TF-IDF retrieval of relevant past context |
| `autourgos-summary-memory` | `SummaryBufferedMemory` | LLM-compressed history to save tokens |
| `autourgos-token-memory` | `TokenBufferedMemory` | Token-budget bounded buffer |
| `autourgos-vector-memory` | `VectorMemory` | Embedding-based recall (you supply the embedding function) |
| `autourgos-episodic-memory` | `EpisodicMemory` | Structured task/outcome log — what was tried, what happened |

---

## Quick Start

`RuntimeShortTermMemory` is soft re-exported from `autourgos_memory` — it only resolves if
`autourgos-buffer-memory` is also installed:

```bash
pip install autourgos-memory autourgos-buffer-memory autourgos-openaichat
```

```python
from autourgos_memory import RuntimeShortTermMemory
from autourgos_agent import Agent
from autourgos_openaichat import OpenAIChatModel

my_llm = OpenAIChatModel(model="gpt-4o-mini")  # needs OPENAI_API_KEY set
memory = RuntimeShortTermMemory(max_messages=20)
agent  = Agent(llm=my_llm, memory=memory)
result = agent.invoke("What did I ask you last time?")
```

---

## Base Interfaces

### MemoryMessage

```python
from autourgos_memory import MemoryMessage
from datetime import datetime, timezone

msg = MemoryMessage(role="user", content="Hello", timestamp=datetime.now(timezone.utc))
print(msg.to_dict())
# {"role": "user", "content": "Hello", "timestamp": "2024-..."}
```

Allowed roles: `user`, `agent`, `system`, `tool`.

### BaseMemory

Implement this to create your own memory backend:

```python
from autourgos_memory import BaseMemory, MemoryMessage

class MyCustomMemory(BaseMemory):
    def add_user_message(self, content: str) -> MemoryMessage: ...
    def add_agent_message(self, content: str) -> MemoryMessage: ...
    def add_tool_message(self, tool_name: str, result: str) -> MemoryMessage: ...
    def format_for_llm(self, query: str = None) -> str: ...
    def clear(self) -> None: ...
```

Only `add_user_message`, `add_tool_message`, and `clear` are true `@abstractmethod`s.
`add_agent_message` and `format_for_llm` are concrete methods with a deprecation-shim fallback:
each one calls through to an older method name (`add_ai_message` / `get_context` respectively)
if your subclass implements *that* one instead, emitting a `DeprecationWarning`. This exists
only to keep a memory backend written against the pre-rename API working unchanged — new
backends should implement `add_agent_message`/`format_for_llm` directly and can ignore
`add_ai_message`/`get_context` entirely.

### BaseRetriever

Implement this to plug in your own vector database:

```python
from autourgos_memory import BaseRetriever, Document

class MyVectorDB(BaseRetriever):
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]: ...
```

### Document

```python
from autourgos_memory import Document

doc = Document(content="Paris is the capital of France.", score=0.92, source="wiki")
```

---

## License

Apache License 2.0, Copyright (c) 2026 Jitin Kumar Sengar
