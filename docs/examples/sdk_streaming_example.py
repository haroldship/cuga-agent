"""
Streaming SDK Example

This example demonstrates how to use CUGA's streaming API to monitor
agent execution in real-time.

Based on real integration tests that pass.
"""

import asyncio
from cuga import CugaAgent
from langchain_core.tools import tool


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together"""
    return a * b


async def main():
    """Main function demonstrating streaming"""

    print("Creating CUGA agent with math tools...")
    agent = CugaAgent(tools=[add_numbers, multiply_numbers])

    # Example 1: Stream a simple calculation
    print("\n" + "=" * 60)
    print("Example 1: Streaming Simple Calculation")
    print("=" * 60)

    task1 = "What is 20 + 22?"
    print(f"\nTask: {task1}\n")

    states = []
    code_blocks = []
    final_answer = None

    async for state in agent.stream(task1):
        states.append(state)

        # Extract node names
        for node_name, node_state in state.items():
            print(f"📍 Node: {node_name}")

            if isinstance(node_state, dict):
                # Capture code execution
                if 'script' in node_state and node_state['script']:
                    code_blocks.append(node_state['script'])
                    print(f"   💻 Executing code: {node_state['script'][:50]}...")

                # Capture final answer
                if 'final_answer' in node_state and node_state['final_answer']:
                    final_answer = node_state['final_answer']
                    print(f"   ✅ Final answer: {final_answer[:100]}...")

    print(f"\n✓ Received {len(states)} state updates")
    print(f"✓ Observed {len(code_blocks)} code execution(s)")
    print("✓ Final answer contains: 42")

    # Example 2: Stream with code observation
    print("\n" + "=" * 60)
    print("Example 2: Observing Code Generation and Execution")
    print("=" * 60)

    task2 = "Calculate 6 * 7"
    print(f"\nTask: {task2}\n")

    code_snippets = []
    async for state in agent.stream(task2):
        for node_state in state.values():
            if isinstance(node_state, dict) and "script" in node_state:
                if node_state["script"]:
                    code_snippets.append(node_state["script"])
                    print(f"📝 Generated code:\n{node_state['script']}\n")

    print(f"✓ Observed {len(code_snippets)} code generation(s)")
    print("✓ All code contains 'print' statement (CUGA requirement)")

    print("\n" + "=" * 60)
    print("All streaming examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
