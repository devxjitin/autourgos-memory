# autourgos-memory

Base memory interfaces for [Autourgos](https://github.com/devxjitin) agents.

This is the **foundation package** — it defines the abstract interfaces (`BaseMemory`, `BaseRetriever`, `MemoryMessage`, `Document`) that all concrete memory implementations use. Install it on its own, or install one of the concrete packages that depend on it.

---

## Install

```bash
# Base interfaces only
pip install autourgos-memory

# Or install concrete implementations individually
pip install autourgos-buffer-memory    # in-memory ring buffer
pip install autourgos-local-memory     # JSON file + SQLite
pip install autourgos-semantic-memory  # TF-IDF keyword retrieval
pip install autourgos-summary-memory   # LLM-compressed rolling summary
pip install autourgos-token-memory     # token-bounded buffer
```

---

## Memory types at a glance

| Package | Class | Best for |
|---|---|---|
| `autourgos-buffer-memory` | `RuntimeShortTermMemory` | Fast in-memory buffer, message-count bounded |
| `autourgos-buffer-memory` | `ConversationBufferMemory` | Unbounded in-memory buffer |
| `autourgos-local-memory` | `LocalShortTermMemory` | Disk persistence via JSON file |
| `autourgos-local-memory` | `SQLiteMemory` | Disk persistence via SQLite, concurrent-safe |
| `autourgos-semantic-memory` | `KeywordMemory` | TF-IDF retrieval of relevant past context |
| `autourgos-summary-memory` | `SummaryBufferedMemory` | LLM-compressed history to save tokens |
| `autourgos-token-memory` | `TokenBufferedMemory` | Token-budget bounded buffer |

---

## Quick start (with concrete packages installed)

`RuntimeShortTermMemory` is soft re-exported from `autourgos_memory` — it only resolves if `autourgos-buffer-memory` is also installed:

```bash
pip install autourgos-memory autourgos-buffer-memory autourgos-openaichat
```

```python
from autourgos_memory import RuntimeShortTermMemory  # requires autourgos-buffer-memory installed
from autourgos_agent import Agent
from autourgos_openaichat import OpenAIChatModel

my_llm = OpenAIChatModel(model="gpt-4o-mini")  # needs OPENAI_API_KEY set
memory = RuntimeShortTermMemory(max_messages=20)
agent  = Agent(llm=my_llm, memory=memory)
result = agent.invoke("What did I ask you last time?")
```

---

## Base interfaces

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

## Links

- PyPI: https://pypi.org/project/autourgos-memory/
- GitHub: https://github.com/devxjitin/autourgos-memory
- Issues: https://github.com/devxjitin/autourgos-memory/issues

---

## License

MIT — see [LICENSE](LICENSE)
