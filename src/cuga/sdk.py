"""
CUGA SDK - Simple interface for using CugaAgent

This module provides a clean, minimal API for using CUGA's agent capabilities.
The agent is built on LangGraph and can be used as a compiled graph or invoked directly.

Example:
    ```python
    from cuga.sdk import CugaAgent
    from langchain_core.tools import tool

    # Define your tools
    @tool
    def search_database(query: str) -> str:
        '''Search the database for information'''
        return "Database results for: " + query

    # Create and run the agent
    agent = CugaAgent(tools=[search_database])
    result = await agent.invoke("Find all users in the database")
    print(result)
    ```
"""

from typing import List, Optional, Dict, Any, Union
from loguru import logger
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler

from cuga.backend.llm.models import LLMManager
from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_lite_graph import (
    create_cuga_lite_graph,
    CugaLiteState,
)
from cuga.backend.cuga_graph.nodes.cuga_lite.direct_langchain_tools_provider import (
    DirectLangChainToolsProvider,
)
from cuga.backend.cuga_graph.nodes.cuga_lite.tool_provider_interface import ToolProviderInterface
from langchain_core.messages import HumanMessage, BaseMessage


class CugaAgent:
    """
    Simple SDK interface for CUGA Agent.

    This class provides a minimal API for creating and invoking CUGA agents.
    Under the hood, it uses LangGraph to create a stateful agent graph.

    Args:
        tools: Optional list of LangChain tools to provide to the agent
        tool_provider: Optional custom tool provider (advanced usage)
        model: Optional language model (defaults to configured model)
        callbacks: Optional list of callback handlers for monitoring

    Attributes:
        graph: The underlying LangGraph StateGraph (compiled)
        tool_provider: The tool provider interface being used

    Example:
        ```python
        from cuga.sdk import CugaAgent
        from langchain_core.tools import tool

        @tool
        def get_weather(city: str) -> str:
            '''Get weather for a city'''
            return f"Weather in {city}: Sunny, 72°F"

        agent = CugaAgent(tools=[get_weather])
        result = await agent.invoke("What's the weather in San Francisco?")
        print(result)  # Agent will use the tool and return an answer
        ```
    """

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        tool_provider: Optional[ToolProviderInterface] = None,
        model: Optional[BaseChatModel] = None,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
    ):
        """
        Initialize the CUGA Agent.

        Args:
            tools: List of LangChain tools (BaseTool or @tool decorated functions)
            tool_provider: Custom tool provider (overrides tools parameter)
            model: Language model to use (defaults to configured model)
            callbacks: List of callback handlers
        """
        self._model = model
        self._callbacks = callbacks
        self._graph = None
        self._compiled_graph = None

        # Setup tool provider
        if tool_provider:
            self.tool_provider = tool_provider
            logger.info("Using custom tool provider")
        elif tools:
            self.tool_provider = DirectLangChainToolsProvider(tools=tools, app_name="runtime_tools")
            logger.info(f"Created DirectLangChainToolsProvider with {len(tools)} tools")
        else:
            self.tool_provider = DirectLangChainToolsProvider(tools=[], app_name="runtime_tools")
            logger.warning("No tools provided - agent will have limited capabilities")

        # Initialize model
        if not self._model:
            from cuga.config import settings

            llm_manager = LLMManager()
            self._model = llm_manager.get_model(settings.agent.code.model)
            logger.info(f"Using default model: {self._model.__class__.__name__}")

    async def _ensure_initialized(self):
        """Ensure tool provider is initialized."""
        if not hasattr(self.tool_provider, 'initialized') or not self.tool_provider.initialized:
            await self.tool_provider.initialize()

    def _create_graph(self, thread_id: Optional[str] = None):
        """Create the LangGraph graph."""
        if self._graph is None:
            self._graph = create_cuga_lite_graph(
                model=self._model,
                tool_provider=self.tool_provider,
                thread_id=thread_id,
                callbacks=self._callbacks,
            )
            logger.debug("Created CugaLite graph")
        return self._graph

    @property
    def graph(self):
        """
        Get the underlying LangGraph StateGraph (compiled).

        This allows advanced users to interact with the graph directly,
        use custom checkpointers, or integrate with LangGraph Cloud.

        Returns:
            Compiled LangGraph graph

        Example:
            ```python
            agent = CugaAgent(tools=[my_tool])
            compiled_graph = agent.graph

            # Use with custom checkpointer
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
            result = await compiled_graph.ainvoke(
                {"chat_messages": [HumanMessage(content="Hello")]},
                config={"configurable": {"thread_id": "user-123"}},
            )
            ```
        """
        if self._compiled_graph is None:
            graph = self._create_graph()
            self._compiled_graph = graph.compile()
            logger.debug("Compiled CugaLite graph")
        return self._compiled_graph

    async def invoke(
        self,
        message: Union[str, List[BaseMessage]],
        thread_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Invoke the agent with a message and get the response.

        This method handles message formatting, graph execution, and response extraction.

        Args:
            message: User message (string) or list of messages for multi-turn conversations
            thread_id: Optional thread ID for E2B sandbox caching
            config: Optional LangGraph config (for advanced usage)

        Returns:
            The agent's final answer as a string

        Example:
            ```python
            # Simple single-turn
            result = await agent.invoke("What's 2+2?")

            # Multi-turn conversation
            messages = [
                HumanMessage(content="My name is Alice"),
                AIMessage(content="Nice to meet you, Alice!"),
                HumanMessage(content="What's my name?"),
            ]
            result = await agent.invoke(messages)
            ```
        """
        await self._ensure_initialized()

        # Convert message to list of BaseMessage
        if isinstance(message, str):
            messages = [HumanMessage(content=message)]
        else:
            messages = message

        # Create initial state with proper CugaLiteState wrapper
        initial_state = CugaLiteState(
            chat_messages=messages,
            thread_id=thread_id,
        )

        # Setup config
        run_config = config or {}
        if thread_id and "configurable" not in run_config:
            run_config["configurable"] = {"thread_id": thread_id}

        # Invoke the graph
        logger.debug(f"Invoking agent with {len(messages)} messages")
        result = await self.graph.ainvoke(initial_state, config=run_config)

        # Extract final answer
        final_answer = result.get("final_answer", "")

        if not final_answer and result.get("error"):
            final_answer = f"Error: {result['error']}"

        return final_answer

    async def stream(
        self,
        message: Union[str, List[BaseMessage]],
        thread_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Stream the agent's execution step by step.

        This method yields state updates as the agent processes the task,
        allowing you to monitor progress in real-time.

        Args:
            message: User message (string) or list of messages
            thread_id: Optional thread ID for E2B sandbox caching
            config: Optional LangGraph config

        Yields:
            State updates as the agent executes

        Example:
            ```python
            async for state in agent.stream("Calculate 10 factorial"):
                print(f"Step: {state.get('step_count', 0)}")
                if state.get('script'):
                    print(f"Code: {state['script']}")
            ```
        """
        await self._ensure_initialized()

        # Convert message to list of BaseMessage
        if isinstance(message, str):
            messages = [HumanMessage(content=message)]
        else:
            messages = message

        # Create initial state with proper CugaLiteState wrapper
        initial_state = CugaLiteState(
            chat_messages=messages,
            thread_id=thread_id,
        )

        # Setup config
        run_config = config or {}
        if thread_id and "configurable" not in run_config:
            run_config["configurable"] = {"thread_id": thread_id}

        # Stream the graph
        logger.debug(f"Streaming agent with {len(messages)} messages")
        async for state in self.graph.astream(initial_state, config=run_config):
            yield state

    def add_tool(self, tool: BaseTool):
        """
        Add a tool to the agent dynamically.

        Note: This only works if using DirectLangChainToolsProvider.
        The graph will need to be recreated on next invocation.

        Args:
            tool: LangChain tool to add

        Example:
            ```python
            agent = CugaAgent(tools=[tool1])

            @tool
            def new_tool(x: int) -> int:
                '''A new tool'''
                return x * 2

            agent.add_tool(new_tool)
            result = await agent.invoke("Use new_tool with 5")
            ```
        """
        if isinstance(self.tool_provider, DirectLangChainToolsProvider):
            self.tool_provider.add_tool(tool)
            # Reset graph so it gets recreated with new tools
            self._graph = None
            self._compiled_graph = None
            logger.info(f"Added tool '{tool.name}' - graph will be recreated on next invocation")
        else:
            raise ValueError(
                "add_tool() only works with DirectLangChainToolsProvider. "
                "Use a custom tool provider for dynamic tool management."
            )

    def add_tools(self, tools: List[BaseTool]):
        """
        Add multiple tools to the agent dynamically.

        Args:
            tools: List of LangChain tools to add

        Example:
            ```python
            agent = CugaAgent()
            agent.add_tools([tool1, tool2, tool3])
            ```
        """
        for tool in tools:
            self.add_tool(tool)


# Convenience function for quick usage
async def run_agent(
    message: str,
    tools: Optional[List[BaseTool]] = None,
    model: Optional[BaseChatModel] = None,
) -> str:
    """
    Convenience function to quickly run an agent with a single message.

    This creates a new agent instance, runs it, and returns the result.
    For multiple invocations, create a CugaAgent instance instead.

    Args:
        message: User message to process
        tools: Optional list of tools
        model: Optional language model

    Returns:
        Agent's response as a string

    Example:
        ```python
        from cuga.sdk import run_agent
        from langchain_core.tools import tool

        @tool
        def calculator(expression: str) -> float:
            '''Evaluate a math expression'''
            return eval(expression)

        result = await run_agent(
            "What's 15 * 23?",
            tools=[calculator]
        )
        print(result)
        ```
    """
    agent = CugaAgent(tools=tools, model=model)
    return await agent.invoke(message)
