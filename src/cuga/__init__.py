"""
CUGA: The Configurable Generalist Agent

CUGA is a state-of-the-art generalist agent designed for enterprise needs,
combining best-of-breed agentic patterns with structured planning and smart
variable management.

Quick Start:
    ```python
    from cuga import CugaAgent
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        '''Get weather for a city'''
        return f"Weather in {city}: Sunny"

    agent = CugaAgent(tools=[get_weather])
    result = await agent.invoke("What's the weather in NYC?")
    print(result.answer)
    ```

For more information, visit: https://cuga.dev
"""

__version__ = "0.2.6"
__all__ = ["CugaAgent", "run_agent", "InvokeResult", "tracked_tool"]


def __getattr__(name):
    if name in ("CugaAgent", "run_agent", "InvokeResult"):
        from cuga.sdk import CugaAgent, run_agent, InvokeResult

        globals()["CugaAgent"] = CugaAgent
        globals()["run_agent"] = run_agent
        globals()["InvokeResult"] = InvokeResult
        return globals()[name]
    if name == "tracked_tool":
        from cuga.backend.cuga_graph.nodes.cuga_lite.tool_call_tracker import tracked_tool

        globals()["tracked_tool"] = tracked_tool
        return tracked_tool
    raise AttributeError(f"module 'cuga' has no attribute {name}")
