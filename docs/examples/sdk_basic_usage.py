"""
Basic SDK Usage Example

This example demonstrates how to use CUGA's Python SDK to create
an agent with custom tools and invoke it with tasks.

All examples in this file are based on real integration tests that pass.
"""

import asyncio
from cuga import CugaAgent
from langchain_core.tools import tool


# Define custom tools using the @tool decorator
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together"""
    return a * b


@tool
def get_greeting(name: str) -> str:
    """Get a greeting for a person"""
    return f"Hello, {name}!"


@tool
def get_user_count() -> int:
    """Get the number of users in the system"""
    return 150


async def main():
    """Main function demonstrating SDK usage"""

    # Create an agent with custom tools
    print("Creating CUGA agent with custom tools...")
    agent = CugaAgent(tools=[add_numbers, multiply_numbers, get_greeting])

    # Example 1: Simple addition
    print("\n" + "=" * 60)
    print("Example 1: Simple Math - Addition")
    print("=" * 60)
    task1 = "What is 10 + 5?"
    print(f"Task: {task1}")
    result1 = await agent.invoke(task1)
    print(f"Result: {result1}")
    print("✓ Agent used add_numbers tool and returned 15")

    # Example 2: Multiplication
    print("\n" + "=" * 60)
    print("Example 2: Simple Math - Multiplication")
    print("=" * 60)
    task2 = "What is 7 * 8?"
    print(f"Task: {task2}")
    result2 = await agent.invoke(task2)
    print(f"Result: {result2}")
    print("✓ Agent used multiply_numbers tool and returned 56")

    # Example 3: Greeting
    print("\n" + "=" * 60)
    print("Example 3: Using Greeting Tool")
    print("=" * 60)
    task3 = "Greet Alice"
    print(f"Task: {task3}")
    result3 = await agent.invoke(task3)
    print(f"Result: {result3}")
    print("✓ Agent used get_greeting tool")

    # Example 4: Multi-step calculation
    print("\n" + "=" * 60)
    print("Example 4: Multi-step Task")
    print("=" * 60)
    task4 = "Calculate (10 + 5) * 3"
    print(f"Task: {task4}")
    result4 = await agent.invoke(task4)
    print(f"Result: {result4}")
    print("✓ Agent used both add_numbers and multiply_numbers tools")

    # Example 5: With thread_id for state isolation
    print("\n" + "=" * 60)
    print("Example 5: Using thread_id for State Isolation")
    print("=" * 60)
    task5 = "What is 20 + 22?"
    print(f"Task: {task5}")
    result5 = await agent.invoke(task5, thread_id="user-session-123")
    print(f"Result: {result5}")
    print("✓ Agent executed with thread_id for E2B caching")

    # Example 6: Dynamic tool addition
    print("\n" + "=" * 60)
    print("Example 6: Adding Tools Dynamically")
    print("=" * 60)
    print("Adding get_user_count tool...")
    agent.add_tool(get_user_count)
    task6 = "How many users are in the system?"
    print(f"Task: {task6}")
    result6 = await agent.invoke(task6)
    print(f"Result: {result6}")
    print("✓ Agent used newly added get_user_count tool")

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
