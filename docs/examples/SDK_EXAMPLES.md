# CUGA SDK Examples

This directory contains examples demonstrating how to use CUGA as a Python SDK in your applications.

## Quick Start

All examples can be run directly with:

```bash
cd docs/examples
uv run python <example_file>.py
```

## Examples Overview

### 1. Basic Usage (`sdk_basic_usage.py`)

**What it demonstrates:**
- Creating a `CugaAgent` with custom tools
- Defining tools using the `@tool` decorator
- Simple task invocation with `agent.invoke()`
- Multiple tool usage in a single task

**Run it:**
```bash
uv run python sdk_basic_usage.py
```

**Key concepts:**
- Tool definition with LangChain's `@tool` decorator
- Agent initialization with a list of tools
- Synchronous-style invocation (async under the hood)

---

### 2. Streaming (`sdk_streaming_example.py`)

**What it demonstrates:**
- Real-time monitoring of agent execution
- Using `agent.stream()` to get step-by-step updates
- Observing code generation and execution
- Tracking agent progress through complex tasks

**Run it:**
```bash
uv run python sdk_streaming_example.py
```

**Key concepts:**
- Streaming API for real-time updates
- State inspection during execution
- Monitoring code execution and outputs

---

### 3. LangGraph Integration (`sdk_langgraph_integration.py`)

**What it demonstrates:**
- Accessing the underlying LangGraph graph
- Using custom checkpointers for state persistence
- Multi-user scenarios with isolated states
- Advanced graph configuration

**Run it:**
```bash
uv run python sdk_langgraph_integration.py
```

**Key concepts:**
- Direct graph access via `agent.graph`
- LangGraph checkpointing for state persistence
- Thread-based state isolation
- Integration with LangGraph ecosystem

---

## SDK API Reference

### `CugaAgent`

Main class for creating and invoking CUGA agents.

```python
from cuga import CugaAgent

agent = CugaAgent(
    tools=[...],          # List of LangChain tools
    tool_provider=None,   # Custom tool provider (advanced)
    model=None,           # Custom LLM (defaults to configured model)
    callbacks=None,       # LangChain callbacks
)
```

**Methods:**

- `async invoke(message, thread_id=None, config=None) -> str`
  - Invoke agent with a message, returns final answer
  
- `async stream(message, thread_id=None, config=None)`
  - Stream agent execution, yields state updates
  
- `add_tool(tool)` / `add_tools(tools)`
  - Dynamically add tools after initialization
  
- `graph` (property)
  - Access the compiled LangGraph StateGraph

### `run_agent()`

Convenience function for one-off invocations.

```python
from cuga import run_agent

result = await run_agent(
    "What's 2+2?",
    tools=[calculator],
    model=None,
)
```

---

## Tool Definition

CUGA uses LangChain's tool system. Define tools using the `@tool` decorator:

```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """Tool description that the LLM sees"""
    return "result"
```

**Best practices:**
- Use clear, descriptive docstrings (the LLM reads these!)
- Add type hints for parameters
- Return strings or JSON-serializable objects
- Keep tools focused on a single responsibility

---

## Configuration

### Environment Variables

Set these in your `.env` file:

```env
# Required: LLM API key
OPENAI_API_KEY=sk-...

# Optional: Model configuration
AGENT_SETTING_CONFIG="settings.openai.toml"
MODEL_NAME=gpt-4o

# Optional: Advanced features
DYNA_CONF_ADVANCED_FEATURES__MODE=api
DYNA_CONF_FEATURES__LOCAL_SANDBOX=true
```

### Model Selection

```python
from langchain_openai import ChatOpenAI
from cuga import CugaAgent

# Use custom model
model = ChatOpenAI(model="gpt-4o", temperature=0)
agent = CugaAgent(tools=[...], model=model)
```

---

## Advanced Usage

### Custom Tool Provider

For advanced scenarios, implement `ToolProviderInterface`:

```python
from cuga.backend.cuga_graph.nodes.cuga_lite.tool_provider_interface import (
    ToolProviderInterface
)

class MyToolProvider(ToolProviderInterface):
    async def get_apps(self): ...
    async def get_tools(self, app_name): ...
    async def get_all_tools(self): ...
    async def initialize(self): ...

agent = CugaAgent(tool_provider=MyToolProvider())
```

### LangGraph Integration

```python
# Access the graph for advanced use cases
graph = agent.graph

# Use with LangGraph Cloud
from langgraph_sdk import get_client
client = get_client(url="http://localhost:8123")
# Deploy graph to LangGraph Cloud...

# Use with custom checkpointer
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver(connection_string="...")
# Recompile graph with checkpointer...
```

---

## Troubleshooting

### Import Errors

If you get import errors, ensure CUGA is installed:

```bash
cd /path/to/cuga-agent
uv sync
```

### Model Configuration

If the agent fails to initialize, check your `.env` file:

```bash
# Verify environment variables are set
cat .env | grep OPENAI_API_KEY
```

### Tool Execution Errors

If tools aren't being called:
- Check tool docstrings are clear and descriptive
- Ensure tool parameters have type hints
- Verify tools return string or JSON-serializable values
- Check logs for security validation errors

---

## More Examples

For more complex examples, see:

- **MCP Integration**: `cuga_as_mcp/` - Using CUGA as an MCP server
- **Runtime Tools**: `cuga_with_runtime_tools/` - Dynamic tool loading
- **Demo Apps**: `demo_apps/` - Full application examples

---

## Support

- **Documentation**: https://cuga.dev
- **GitHub**: https://github.com/cuga-project/cuga-agent
- **Discord**: https://discord.gg/aH6rAEEW

---

## License

Apache 2.0 - See [LICENSE](../../LICENSE) for details.

