# Agentic Architecture Patterns

A reference architecture collection demonstrating agentic AI design patterns implemented across **Strands Agents SDK**, **LangGraph**, and **CrewAI**. Built for architects and engineers evaluating agent frameworks and patterns.

## Patterns

| # | Pattern | Description | Best For |
|---|---------|-------------|----------|
| 01 | [ReAct](patterns/01-react/) | Reasoning + Acting loop with tool use | General-purpose tool-using agents |
| 02 | [Plan-and-Execute](patterns/02-plan-and-execute/) | Separate planning from execution | Complex multi-step tasks |
| 03 | [Multi-Agent Supervisor](patterns/03-multi-agent-supervisor/) | Supervisor delegates to specialist agents | Domain-diverse workloads |
| 04 | [Human-in-the-Loop](patterns/04-human-in-the-loop/) | Agent pauses for human approval | High-stakes decisions |
| 05 | [RAG Agent](patterns/05-rag-agent/) | Retrieval-augmented generation with agent routing | Knowledge-intensive tasks |

## Framework Comparison Matrix

| Capability | Strands Agents | LangGraph | CrewAI |
|------------|---------------|-----------|--------|
| **Paradigm** | SDK/imperative | Graph-based state machine | Declarative role-based |
| **Orchestration** | Agent loop with tools | Nodes + edges + state | Crew with process type |
| **State Management** | Conversation context | Explicit TypedDict state | Shared crew memory |
| **Human-in-the-Loop** | Custom tool pattern | First-class interrupt | Human input tool |
| **Multi-Agent** | Agent-as-tool composition | Supervisor/swarm graphs | Crew with manager |
| **Streaming** | Built-in event hooks | Stream modes (values/updates) | Callback handlers |
| **Persistence** | Custom implementation | Checkpointer (SQLite/Postgres) | Built-in memory types |
| **Best For** | AWS-native, Bedrock-first apps | Complex stateful workflows | Rapid prototyping, role-play agents |

## Architecture Decision Records

See [docs/decision-tree.md](docs/decision-tree.md) for guidance on selecting the right pattern and framework for your use case.

See [docs/framework-comparison.md](docs/framework-comparison.md) for detailed trade-off analysis.

## Getting Started

```bash
pip install -r requirements.txt

# Set up AWS credentials for Strands (uses Bedrock)
export AWS_PROFILE=bedrock
export AWS_REGION=us-west-2

# Set up OpenAI key for LangGraph/CrewAI examples
export OPENAI_API_KEY=your-key-here
```

## Project Structure

```
agentic-architecture-patterns/
├── patterns/
│   ├── 01-react/              # ReAct pattern
│   ├── 02-plan-and-execute/   # Plan-and-Execute pattern
│   ├── 03-multi-agent-supervisor/  # Supervisor pattern
│   ├── 04-human-in-the-loop/  # HITL pattern
│   └── 05-rag-agent/          # RAG-augmented agent
├── docs/
│   ├── framework-comparison.md
│   └── decision-tree.md
├── requirements.txt
└── README.md
```

## License

MIT
