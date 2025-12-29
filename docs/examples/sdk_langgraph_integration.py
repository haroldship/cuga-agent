"""
LangGraph Integration Example

This example demonstrates how to access CUGA's underlying LangGraph
for advanced use cases like custom checkpointing and state management.
"""

import asyncio
from cuga import CugaAgent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver


@tool
def save_note(note: str) -> str:
    """Save a note to memory"""
    return f"Note saved: {note}"


@tool
def get_time() -> str:
    """Get current time"""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def example_basic_graph_access():
    """Example 1: Access the compiled LangGraph"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Graph Access")
    print("=" * 60)

    agent = CugaAgent(tools=[save_note, get_time])

    # Access the underlying compiled LangGraph
    graph = agent.graph
    print(f"Graph type: {type(graph)}")
    print("The graph is a compiled LangGraph StateGraph that can be:")
    print("  - Deployed to LangGraph Cloud")
    print("  - Used with custom checkpointers")
    print("  - Integrated into larger LangGraph applications")

    # Use it directly
    initial_state = {
        "chat_messages": [HumanMessage(content="What time is it?")],
    }
    result = await graph.ainvoke(initial_state)
    print(f"\nDirect graph invocation result: {result.get('final_answer', 'No answer')}")


async def example_with_checkpointer():
    """Example 2: Use graph with custom checkpointer for state persistence"""
    print("\n" + "=" * 60)
    print("Example 2: Graph with Checkpointer")
    print("=" * 60)

    agent = CugaAgent(tools=[save_note, get_time])

    # Create a checkpointer for state persistence
    checkpointer = MemorySaver()

    # Get the graph and recompile with checkpointer
    # Note: For production, you'd want to integrate this at graph creation time
    from cuga.backend.cuga_graph.nodes.cuga_lite.cuga_lite_graph import (
        create_cuga_lite_graph,
    )
    from cuga.backend.llm.models import LLMManager

    llm_manager = LLMManager()
    model = llm_manager.get_model()

    graph = create_cuga_lite_graph(
        model=model,
        tool_provider=agent.tool_provider,
    )

    # Compile with checkpointer
    compiled_graph = graph.compile(checkpointer=checkpointer)

    # First conversation
    print("\nFirst conversation (thread_id='user-123'):")
    state1 = {
        "chat_messages": [HumanMessage(content="Save a note: Meeting at 3pm")],
    }
    result1 = await compiled_graph.ainvoke(
        state1,
        config={"configurable": {"thread_id": "user-123"}},
    )
    print(f"Result: {result1.get('final_answer', 'No answer')[:100]}...")

    # Second conversation - same thread
    print("\nSecond conversation (same thread_id='user-123'):")
    state2 = {
        "chat_messages": [HumanMessage(content="What time is it now?")],
    }
    result2 = await compiled_graph.ainvoke(
        state2,
        config={"configurable": {"thread_id": "user-123"}},
    )
    print(f"Result: {result2.get('final_answer', 'No answer')[:100]}...")

    print("\nNote: State is persisted across invocations with the same thread_id")


async def example_multi_user():
    """Example 3: Multi-user scenario with isolated states"""
    print("\n" + "=" * 60)
    print("Example 3: Multi-User State Isolation")
    print("=" * 60)

    agent = CugaAgent(tools=[save_note])

    # User 1
    print("\nUser 1 (thread_id='alice'):")
    result1 = await agent.invoke("Save a note: Alice's project deadline is Friday", thread_id="alice")
    print(f"Result: {result1[:100]}...")

    # User 2
    print("\nUser 2 (thread_id='bob'):")
    result2 = await agent.invoke("Save a note: Bob's meeting is tomorrow", thread_id="bob")
    print(f"Result: {result2[:100]}...")

    print("\nNote: Each user has isolated state via thread_id")


async def main():
    """Run all examples"""
    print("=" * 60)
    print("CUGA LangGraph Integration Examples")
    print("=" * 60)

    await example_basic_graph_access()
    await example_with_checkpointer()
    await example_multi_user()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
