# autourgos-memory

[![Framework: Autourgos](https://img.shields.io/badge/Framework-Autourgos-orange.svg)](https://github.com/devxjitin)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://pypi.org/project/autourgos-memory/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/devxjitin/autourgos-memory/blob/main/LICENSE)
[![Author](https://img.shields.io/badge/Author-Jitin%20Kumar%20Sengar-blue.svg)](https://github.com/devxjitin)
[![Contributor](https://img.shields.io/badge/Contributor-Sonia-blueviolet.svg)](https://github.com/dahiyasonia)
[![Contributor](https://img.shields.io/badge/Contributor-Vishwanil%20Suman-blueviolet.svg)]()

Unified memory package for [Autourgos](https://github.com/devxjitin) agents. One package,
one install, all seven memory backends — plus the abstract interfaces (`BaseMemory`, `BaseRetriever`,
`MemoryMessage`, `Document`) they all implement.

```python
from autourgos_memory import RuntimeShortTermMemory
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
- **Every concrete backend, bundled**: `RuntimeShortTermMemory`, `ConversationBufferMemory`,
  `ExpiringBufferMemory`, `LocalShortTermMemory`, `SQLiteMemory`, `KeywordRetriever`, `KeywordMemory`,
  `SummaryBufferedMemory`, `TokenBufferedMemory`, `VectorMemory`, `VectorRetriever`, `Episode`,
  `EpisodicMemory` — all import directly from `autourgos_memory`, no sibling packages to install
- Only `VectorMemory`/`VectorRetriever` (needs `numpy`) and accurate `TokenBufferedMemory` counting (needs
  `tiktoken`) pull in extra dependencies — install via the `[vector]`/`[tiktoken]` extras

---

## Table of Contents

- [Install](#install)
- [Memory Types at a Glance](#memory-types-at-a-glance)
- [Quick Start](#quick-start)
- [Base Interfaces](#base-interfaces)
- [Memory Types Explained](#memory-types-explained)
  - [RuntimeShortTermMemory](#runtimeshorttermmemory)
  - [ConversationBufferMemory](#conversationbuffermemory)
  - [ExpiringBufferMemory](#expiringbuffermemory)
  - [LocalShortTermMemory](#localshorttermmemory)
  - [SQLiteMemory](#sqlitememory)
  - [KeywordMemory / KeywordRetriever](#keywordmemory--keywordretriever)
  - [SummaryBufferedMemory](#summarybufferedmemory)
  - [TokenBufferedMemory](#tokenbufferedmemory)
  - [VectorMemory / VectorRetriever](#vectormemory--vectorretriever)
  - [EpisodicMemory](#episodicmemory)
- [License](#license)

---

## Install

```bash
pip install autourgos-memory              # everything except VectorMemory and tiktoken counting
pip install autourgos-memory[vector]      # + VectorMemory/VectorRetriever (numpy)
pip install autourgos-memory[tiktoken]    # + accurate TokenBufferedMemory counting
pip install autourgos-memory[all]         # both extras
```

---

## Memory Types at a Glance

| Class | Module | Best for |
|---|---|---|
| `RuntimeShortTermMemory` | `autourgos_memory.buffer` | Fast in-memory buffer, message-count bounded |
| `ConversationBufferMemory` | `autourgos_memory.buffer` | Unbounded in-memory buffer |
| `ExpiringBufferMemory` | `autourgos_memory.buffer` | Run-scoped facts that expire after a TTL |
| `LocalShortTermMemory` | `autourgos_memory.local` | Disk persistence via JSON file |
| `SQLiteMemory` | `autourgos_memory.local` | Disk persistence via SQLite, concurrent-safe |
| `KeywordMemory` | `autourgos_memory.semantic` | TF-IDF retrieval of relevant past context |
| `SummaryBufferedMemory` | `autourgos_memory.summary` | LLM-compressed history to save tokens |
| `TokenBufferedMemory` | `autourgos_memory.token` | Token-budget bounded buffer |
| `VectorMemory` | `autourgos_memory.vector` | Embedding-based recall (you supply the embedding function); needs `[vector]` |
| `EpisodicMemory` | `autourgos_memory.episodic` | Structured task/outcome log — what was tried, what happened |

Every class above is also exported directly from the top-level `autourgos_memory` package.

---

## Quick Start

```bash
pip install autourgos-memory autourgos-openaichat
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
from datetime import datetime
from typing import Optional
from autourgos_memory import BaseMemory, MemoryMessage

class MyCustomMemory(BaseMemory):
    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None) -> MemoryMessage: ...
    def format_for_llm(self, query: str = None) -> str: ...
    def clear(self) -> None: ...
```

Only `add_message`, `format_for_llm`, and `clear` are true `@abstractmethod`s.
`add_user_message`/`add_agent_message`/`add_system_message`/`add_tool_message` are concrete
defaults built on `add_message(role, content, timestamp=None)` — implement `add_message` once
and every backend gets all four for free. `add_agent_message` and `format_for_llm` also carry a
deprecation-shim fallback: each calls through to an older method name (`add_ai_message` /
`get_context` respectively) if your subclass implements *that* one instead, emitting a
`DeprecationWarning`. This exists only to keep a memory backend written against the pre-rename
API working unchanged.

### format_conversation_banner / ROLE_TO_OPENAI

Shared helpers for backends implementing `format_for_llm`/`get_messages`:

```python
from autourgos_memory import format_conversation_banner, ROLE_TO_OPENAI

format_conversation_banner(messages, include_timestamps=True)
# "\n--- Previous Conversation Context ---\n[2026-...] user: hi\n--------------------------------------\n"

ROLE_TO_OPENAI  # {"user": "user", "agent": "assistant", "system": "system", "tool": "tool"}
```

### RetrievalAugmentedMemory

Base class for a dual-store memory: a short-term buffer (recent turns, always included) plus a
`BaseRetriever` (older, relevant turns, surfaced only when a query is given). This is the shared
shape behind `KeywordMemory` (`autourgos_memory.semantic`) and `VectorMemory`
(`autourgos_memory.vector`) — subclass it, build your own `retriever`, and pass both to `super().__init__()`:

```python
from autourgos_memory import RetrievalAugmentedMemory

class MyRetrievalMemory(RetrievalAugmentedMemory):
    def __init__(self, my_retriever, short_term=None, top_k=3):
        super().__init__(short_term=short_term or MyShortTermMemory(), retriever=my_retriever, top_k=top_k)
```

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

## Memory Types Explained

What each backend is actually for, when to reach for it, every constructor parameter, and a runnable demo.

### RuntimeShortTermMemory

Module: `autourgos_memory.buffer`

**Use when** you just need the last N turns in RAM for a single process's lifetime — no persistence, no
retrieval, cheapest possible option. Caps by message count; oldest messages are dropped first once the cap
is exceeded.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_messages` | `int` | `20` | Maximum messages kept. Once exceeded, the oldest are dropped so only the most recent `max_messages` remain. Must be `>= 1`. |
| `name` | `str` | `"runtime"` | Human-readable identifier for this memory instance (useful when an agent juggles more than one). |

```python
from autourgos_memory import RuntimeShortTermMemory

memory = RuntimeShortTermMemory(max_messages=20, name="chat")
memory.add_user_message("My favorite color is blue.")
memory.add_agent_message("Got it, blue it is!")
print(memory.format_for_llm())
# --- Previous Conversation Context ---
# user: My favorite color is blue.
# agent: Got it, blue it is!
# --------------------------------------
```

### ConversationBufferMemory

Module: `autourgos_memory.buffer` (subclasses `RuntimeShortTermMemory`)

**Use when** you want to keep *every* message in RAM for the session, with no truncation at all. For long
conversations, prefer `RuntimeShortTermMemory` with a cap, or `SummaryBufferedMemory` for LLM compression.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"conversation"` | Human-readable identifier. Internally this class just calls `RuntimeShortTermMemory.__init__(max_messages=sys.maxsize, name=name)`, so there's no `max_messages` to set — it's effectively unbounded. |

```python
from autourgos_memory import ConversationBufferMemory

memory = ConversationBufferMemory(name="full-history")
memory.add_user_message("Let's plan the Q3 roadmap.")
memory.add_agent_message("Sure — what are the top priorities?")
print(memory.format_for_llm())
```

### ExpiringBufferMemory

Module: `autourgos_memory.buffer`

**Use when** a long-running (possibly background) agent needs temporary, run-scoped facts — things worth
remembering for the next few minutes or hours of a task, but that shouldn't silently persist into a later,
unrelated run. Expired messages are purged lazily (on the next add/read call, no background thread) and
never appear in `get_messages()`/`format_for_llm()`.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `default_ttl_seconds` | `float \| None` | `None` | Time-to-live applied to a message when `add_message()`/`add_user_message()` etc. don't pass their own `ttl_seconds`. `None` means messages never expire unless a per-call `ttl_seconds` is given. Must be `> 0` if set. |
| `max_messages` | `int \| None` | `None` | Ring-buffer cap on *live* (non-expired) messages. `None` is unbounded. Must be `>= 1` if set. |
| `name` | `str` | `"expiring"` | Human-readable identifier. |

Every `add_*_message()` method also takes an optional per-call `ttl_seconds: float | None`, which overrides
`default_ttl_seconds` for that one message.

```python
from autourgos_memory import ExpiringBufferMemory

memory = ExpiringBufferMemory(default_ttl_seconds=300, max_messages=50)
memory.add_user_message("The deploy is currently paused for maintenance.")
memory.add_user_message("This fact should live longer.", ttl_seconds=3600)  # per-call override
print(memory.format_for_llm())
# both messages appear now; the first one silently drops out after 300s
```

### LocalShortTermMemory

Module: `autourgos_memory.local`

**Use when** conversation history needs to survive a process restart but you don't need a real database
server — a CLI tool, a desktop app, a single-machine service. Writes a JSON file (simple, human-readable,
fine for light traffic); thread-safe via a file-level lock, and uses an atomic write (tmp file → replace) to
prevent corruption. Safe for multiple processes reading the same file.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file_path` | `str` | `"./data/local_memory.json"` | Path to the JSON memory file. Parent folders are created automatically. |
| `max_messages` | `int` | `20` | Rolling cap — oldest messages are pruned after each write. Must be `>= 1`. |
| `name` | `str` | `"local"` | Human-readable identifier. |
| `create_if_missing` | `bool` | `True` | If the file doesn't exist yet, create it (as an empty list) immediately in `__init__` rather than waiting for the first `add_message()`. |
| `lock_timeout_seconds` | `float` | `10.0` | How long to wait when acquiring the file lock before raising `TimeoutError` (guards against a stale lock from a crashed process). |

```python
from autourgos_memory import LocalShortTermMemory

memory = LocalShortTermMemory(file_path="./data/session.json", max_messages=50)
memory.add_user_message("Remind me to call the client tomorrow.")
# ... process restarts here ...
memory2 = LocalShortTermMemory(file_path="./data/session.json")
print(memory2.get_messages()[-1].content)
# "Remind me to call the client tomorrow."
```

### SQLiteMemory

Module: `autourgos_memory.local`

**Use when** you want disk persistence but need it safer under concurrent writes and cheaper to query at
scale than a JSON file — SQLite in WAL mode, no external lock file needed. Prefer this over
`LocalShortTermMemory` once history gets large or multiple writers are involved.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `db_path` | `str` | `"./data/autourgos_memory.db"` | Path to the `.db` file. Use `":memory:"` for an ephemeral in-process database. |
| `max_messages` | `int \| None` | `500` | Rolling cap — oldest rows are evicted once exceeded. `None` = unlimited. |
| `name` | `str` | `"sqlite"` | Human-readable identifier. |

`get_messages()` also accepts an optional `limit: int | None` to fetch only the most recent N messages
instead of the full history. `SQLiteMemory` supports the context-manager protocol (`with SQLiteMemory(...) as m:`) and a `close()` method to release the connection explicitly.

```python
from autourgos_memory import SQLiteMemory

with SQLiteMemory(db_path="chat_history.db", max_messages=200) as memory:
    memory.add_user_message("Remind me to call the client tomorrow.")
    print(memory.get_messages(limit=1)[-1].content)
    # "Remind me to call the client tomorrow."
```

### KeywordMemory / KeywordRetriever

Module: `autourgos_memory.semantic`

**Use when** you want the agent to recall relevant *older* turns (not just the last N) based on keyword
overlap, without paying for an embedding model or vector database — a good zero-dependency middle ground
before reaching for `VectorMemory`. Good for FAQ-style or support-ticket agents where users often re-ask
about the same named topics. `KeywordMemory` is a dual-store: a sliding short-term buffer (always included)
plus a `KeywordRetriever` (TF-IDF cosine similarity, older turns surfaced only when a query is given).

**`KeywordRetriever` parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_documents` | `int \| None` | `None` | Oldest indexed documents are FIFO-evicted once the corpus exceeds this count. `None` keeps everything. |

**`KeywordMemory` parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `short_term` | `BaseMemory \| None` | `None` | The recent-turns buffer. Defaults to `RuntimeShortTermMemory(max_messages=10, name="keyword")`. Pass your own to change its capacity or backend. |
| `retriever` | `BaseRetriever \| None` | `None` | The long-term recall store. Defaults to a fresh `KeywordRetriever(max_documents=max_documents)`. |
| `top_k` | `int` | `3` | How many relevant past documents to surface per query, in `format_for_llm(query=...)`. |
| `max_documents` | `int \| None` | `None` | Forwarded to the default `KeywordRetriever` when `retriever` is not given directly; ignored if you pass your own `retriever`. |

```python
from autourgos_memory import KeywordMemory

memory = KeywordMemory(top_k=2)
memory.add_user_message("Our production deploy uses us-east-1.")
for i in range(15):
    memory.add_user_message(f"unrelated filler message {i}")  # pushes the deploy turn out of the buffer

print(memory.format_for_llm(query="which region is the deploy in?"))
# --- Relevant Past Context ---
# [user]: Our production deploy uses us-east-1.
# -----------------------------
# ...recent buffer follows...
```

### SummaryBufferedMemory

Module: `autourgos_memory.summary`

**Use when** conversations run long enough that raw history would blow the context window, and you have an
LLM available to compress it. Keeps the last `max_messages` in full; older turns get folded into a rolling
`moving_summary` (LLM-compressed if an `llm` is given, raw-concatenated as a fallback otherwise, or if the
LLM call fails).

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `llm` | `Any \| None` | `None` | An object with an `.invoke(prompt: str)` method (e.g. `OpenAIChatModel`). If omitted, older history is concatenated verbatim instead of LLM-compressed. |
| `max_messages` | `int` | `10` | Number of most-recent messages kept in full before the overflow gets folded into `moving_summary`. Must be `>= 1`. |
| `moving_summary` | `str` | `""` | Seed summary to start with, if you already have one (e.g. resuming a session). |
| `max_summary_chars` | `int \| None` | `None` | Caps `moving_summary`'s length on the raw-concatenation fallback path only (no `llm`, or LLM compression failed) — oldest content is trimmed from the front once exceeded. `None` disables the cap. Never affects a successfully LLM-compressed summary. |

```python
from autourgos_memory import SummaryBufferedMemory

# llm needs an .invoke(prompt: str) method (e.g. an OpenAIChatModel, see Quick Start above);
# without one, moving_summary falls back to raw concatenation instead of LLM compression.
memory = SummaryBufferedMemory(llm=my_llm, max_messages=5)
for i in range(20):
    memory.add_user_message(f"turn {i}: discussing the Q3 roadmap")

print(memory.format_for_llm())
# --- Summary of Past Conversation ---
# <LLM-compressed summary of turns 0-14>
# ------------------------------------
#
# --- Recent Conversation Context ---
# [...] user: turn 15: discussing the Q3 roadmap
# ...
# [...] user: turn 19: discussing the Q3 roadmap
# -----------------------------------
```

### TokenBufferedMemory

Module: `autourgos_memory.token`

**Use when** you're budgeting by tokens rather than message count — e.g. packing as much history as
possible into a fixed context window without a compression step. Falls back to a character-based heuristic
without `tiktoken`; install the `[tiktoken]` extra for accurate counts.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_tokens` | `int` | `2000` | Token budget for the whole buffer. Oldest messages are evicted from the front whenever the running total exceeds this. Must be `>= 1`. |
| `token_estimator` | `Callable[[str], int] \| None` | `None` | Custom `(text) -> token_count` function. Defaults to `tiktoken`'s `cl100k_base` encoding if installed, otherwise a Unicode-aware character heuristic. |
| `oversized_message_policy` | `str` | `"keep"` | What to do with a single new message whose own token cost already exceeds `max_tokens` (so evicting everything else still wouldn't free enough room). One of: `"keep"` (store it anyway and log a warning — the cap is knowingly violated for it), `"truncate"` (cut the content down to fit, appending a `[TRUNCATED]` marker), `"drop"` (discard the message entirely and log a warning). |

`total_tokens` (read-only property) reports the current running token count.

```python
from autourgos_memory import TokenBufferedMemory

memory = TokenBufferedMemory(max_tokens=500, oversized_message_policy="truncate")
memory.add_user_message("Here is a long paste of logs...")
print(memory.total_tokens, "/", memory.max_tokens)
```

### VectorMemory / VectorRetriever

Module: `autourgos_memory.vector` — requires the `[vector]` extra (`numpy`)

**Use when** you need real semantic similarity search (not just keyword overlap) — a support bot recalling
paraphrased past questions, a research assistant over a large document set. You supply the `embed_fn`
(local model or a cloud embeddings API) — this module only stores vectors (SQLite-persisted) and ranks by
cosine similarity; it never computes embeddings itself.

**`VectorRetriever` parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `embed_fn` | `Callable[[str], Sequence[float]]` | *required* | Called once per document on `add_document()` and once per `retrieve()` call to embed the query. Wraps a local model or a cloud API call. |
| `db_path` | `str` | `":memory:"` | SQLite file path. `":memory:"` keeps everything in RAM and discards it on process exit; pass a real path for persistence across restarts. |
| `max_documents` | `int \| None` | `None` | Oldest documents are dropped once this count is exceeded. `None` keeps everything. |

**`VectorMemory` parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `embed_fn` | `Callable[[str], Sequence[float]] \| None` | `None` | Required *unless* you pass a pre-built `retriever`. Used to build the default `VectorRetriever`. |
| `short_term` | `BaseMemory \| None` | `None` | The recent-turns buffer. Defaults to `RuntimeShortTermMemory(max_messages=10, name="vector")`. |
| `retriever` | `VectorRetriever \| None` | `None` | Pass a pre-built `VectorRetriever` instead of letting `VectorMemory` construct one from `embed_fn`/`db_path`/`max_documents`. |
| `db_path` | `str` | `":memory:"` | Forwarded to the default `VectorRetriever` when `retriever` is not given directly. |
| `top_k` | `int` | `3` | How many relevant past documents to surface per query. |
| `max_documents` | `int \| None` | `None` | Forwarded to the default `VectorRetriever` when `retriever` is not given directly. |

```python
from autourgos_memory import VectorMemory

def embed(text: str) -> list[float]:
    ...  # call your local model or a cloud embeddings API

memory = VectorMemory(embed_fn=embed, db_path="memory.db", top_k=1)
memory.add_user_message("My favorite color is blue.")
for i in range(15):
    memory.add_user_message(f"filler message {i}")  # pushes the color turn out of the buffer

print(memory.format_for_llm(query="what color do I like?"))
# --- Relevant Past Context ---
# [user]: My favorite color is blue.
# -----------------------------
# ...recent buffer follows...
```

### EpisodicMemory

Module: `autourgos_memory.episodic`

**Use when** the agent is autonomous and repeats tasks over time — you want it to recall *what was tried
and what happened*, not conversation turns. Persisted to SQLite, retrieval reuses `KeywordMemory`'s TF-IDF
scoring. Good for a coding agent, ops runbook, or any loop that should avoid repeating a known-bad approach.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `db_path` | `str` | `":memory:"` | SQLite file path. `":memory:"` keeps everything in RAM and discards it on process exit; pass a real path for recall across restarts. |
| `max_episodes` | `int \| None` | `None` | Oldest episodes are dropped once this count is exceeded. `None` keeps everything. |

`remember(task, outcome, actions_taken=None, notes="")` records one episode:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | `str` | *required* | Non-empty description of what was attempted. |
| `outcome` | `str` | *required* | One of `"success"`, `"fail"`, `"partial"`. |
| `actions_taken` | `list[str] \| None` | `None` | Ordered list of steps/actions performed. |
| `notes` | `str` | `""` | Freeform notes (e.g. why it failed, what to try next). |

```python
from autourgos_memory import EpisodicMemory

memory = EpisodicMemory(db_path="episodes.db")
memory.remember(
    task="Deploy the app to production",
    outcome="fail",
    actions_taken=["ran deploy.sh", "hit S3 permission error"],
    notes="Needs IAM role update before retrying.",
)

for ep in memory.retrieve("deploying the app", top_k=3):
    print(ep.content)
# Task: Deploy the app to production
# Outcome: fail
# Actions taken: ran deploy.sh; hit S3 permission error
# Notes: Needs IAM role update before retrying.
```

---

## License

Apache License 2.0, Copyright (c) 2026 Jitin Kumar Sengar
