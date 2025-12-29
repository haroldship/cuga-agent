"""
CugaLite LangGraph - Unified subgraph combining CugaAgent and CodeAct

TODO: Multi-user, multi-model, multi-tools dependency injection refactoring
----------------------------------------------------------------------------
CURRENT STATE: Supports multi-user but with same configuration (shared model, tools, memory backend)
GOAL: Enable per-user configuration with isolated models, tools, and memory

This class needs architectural changes to support multi-user, multi-model, and multi-tools scenarios:

1. Multi-Tools Client Dependency Injection:
   - Replace direct tool_provider parameter with injectable tools_client per user_id
   - Use LangGraph's configurable pattern to inject tools_client at runtime
   - Store tools_client in graph config/state rather than closure
   - Enable per-user tool access control and isolation
   - Support dynamic tool loading/unloading per user session
   - Allow different tool sets per user based on permissions/subscriptions

2. Multi-Model Support:
   - Replace hardcoded model parameter with injectable model per user/session
   - Support per-user model selection (different LLMs for different users)
   - Enable model switching mid-conversation via configurable
   - Inject llm_manager or model_client rather than direct model instance
   - Support model-specific configurations (temperature, max_tokens, etc.) per user

3. Multi-User Memory/Storage:
   - Implement user_id-scoped memory storage (variables_storage, chat_messages)
   - Use injectable memory backend (e.g., per-user checkpointer)
   - Migrate from global state to user-scoped state partitioning
   - Consider LangGraph's built-in checkpointing with user_id as partition key
   - Isolate conversation history and variables per user

4. LangGraph Configurable Dependencies:
   - Leverage config["configurable"] for runtime dependency injection
   - Pass user_id, tools_client, model_client, memory_backend via configurable
   - Remove hardcoded global instances (tracker, llm_manager)
   - Make all external dependencies (LLM, tools, memory) injectable
   - Support per-request configuration overrides

5. State Isolation:
   - Ensure CugaLiteState is scoped per user session
   - Add user_id field to state for tracking and isolation
   - Implement proper state cleanup between user sessions
   - Consider multi-tenancy patterns for shared resources
   - Track model_id and tools_version in state for debugging

6. Thread Safety:
   - Ensure thread-safe access to user-scoped resources
   - Avoid shared mutable state between users
   - Use proper async/await patterns for concurrent user requests
   - Handle concurrent model/tool requests from different users
"""

import re
import json
from typing import Any, Optional, Sequence, Dict, List, Tuple
from loguru import logger
from pydantic import BaseModel, Field

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from cuga.backend.activity_tracker.tracker import ActivityTracker, Step
from cuga.backend.llm.models import LLMManager
from cuga.backend.cuga_graph.state.agent_state import AgentState
from cuga.backend.cuga_graph.nodes.cuga_lite.prompt_utils import create_mcp_prompt, PromptUtils
from cuga.backend.cuga_graph.nodes.cuga_lite.executors import CodeExecutor
from cuga.backend.cuga_graph.nodes.cuga_lite.tool_provider_interface import ToolProviderInterface
from cuga.config import settings
from cuga.configurations.instructions_manager import get_all_instructions_formatted
from cuga.backend.llm.utils.helpers import load_one_prompt
from pathlib import Path

try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
except ImportError:
    try:
        from langfuse.callback.langchain import LangchainCallbackHandler as LangfuseCallbackHandler
    except ImportError:
        LangfuseCallbackHandler = None


tracker = ActivityTracker()
llm_manager = LLMManager()

BACKTICK_PATTERN = r'```python(.*?)```'


class CugaLiteState(BaseModel):
    """State for CugaLite subgraph.

    Shared keys with AgentState:
    - chat_messages: List[BaseMessage] (primary message history)
    - final_answer: str (compatible with parent)
    - variables_storage: Dict[str, Dict[str, Any]] (shared variables)
    - variable_counter_state: int (shared variable counter)
    - variable_creation_order: List[str] (shared variable order)
    - cuga_lite_metadata: Dict[str, Any] (metadata for tracking execution context)
    - sub_task: str (current subtask being executed)
    - sub_task_app: str (app name for subtask)
    - api_intent_relevant_apps: List[Any] (relevant apps for the task)

    Subgraph-only keys:
    - script, execution_complete, error, metrics
    - tools_prepared: bool (flag indicating tools have been prepared)
    - prepared_prompt: str (dynamically generated prompt)
    """

    # Shared keys (compatible with AgentState)
    chat_messages: Optional[List[BaseMessage]] = Field(default_factory=list)
    final_answer: Optional[str] = ""
    thread_id: Optional[str] = None
    variables_storage: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    variable_counter_state: int = 0
    variable_creation_order: List[str] = Field(default_factory=list)
    cuga_lite_metadata: Optional[Dict[str, Any]] = None
    sub_task: Optional[str] = None
    sub_task_app: Optional[str] = None
    api_intent_relevant_apps: Optional[List[Any]] = None

    # Subgraph-only keys
    tools_prepared: bool = False
    prepared_prompt: Optional[str] = None
    script: Optional[str] = None
    execution_complete: bool = False
    error: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    step_count: int = 0  # Counter for number of steps (call_model + sandbox cycles)

    class Config:
        arbitrary_types_allowed = True

    @property
    def variables_manager(self):
        """Get a state-specific variables manager that stores data in this CugaLiteState.

        Uses the same StateVariablesManager as AgentState for consistent interface.
        """
        from cuga.backend.cuga_graph.state.agent_state import StateVariablesManager

        return StateVariablesManager(self)


def extract_and_combine_codeblocks(text: str) -> str:
    """Extract all codeblocks from a text string and combine them."""
    code_blocks = re.findall(BACKTICK_PATTERN, text, re.DOTALL)

    if code_blocks:
        processed_blocks = []
        for block in code_blocks:
            block = block.strip()
            processed_blocks.append(block)

        combined_code = "\n\n".join(processed_blocks)

        if "print(" not in combined_code:
            return ""

        return combined_code

    stripped_text = text.strip()

    if "print(" not in stripped_text:
        return ""

    try:
        compile(stripped_text.replace('await ', ''), '<string>', 'exec')
        return stripped_text
    except SyntaxError:
        return ""


def append_chat_messages_with_step_limit(
    state: CugaLiteState, new_messages: List[BaseMessage]
) -> Tuple[List[BaseMessage], Optional[AIMessage]]:
    """Append new messages to chat_messages with step counting and limit checking.

    Args:
        state: Current CugaLiteState
        new_messages: List of new messages to append

    Returns:
        Tuple of (updated_chat_messages, error_message)
        - updated_chat_messages: Updated list of chat messages
        - error_message: AIMessage with error if limit reached, None otherwise
    """
    max_steps = settings.advanced_features.cuga_lite_max_steps
    new_step_count = state.step_count + 1

    if new_step_count > max_steps:
        error_msg = (
            f"Maximum step limit ({max_steps}) reached. "
            f"The task has exceeded the allowed number of execution cycles. "
            f"Please simplify your request or break it into smaller tasks."
        )
        logger.warning(f"Step limit reached: {new_step_count} > {max_steps}")
        error_ai_message = AIMessage(content=error_msg)
        return state.chat_messages + new_messages + [error_ai_message], error_ai_message

    logger.debug(f"Step count: {new_step_count}/{max_steps}")
    return state.chat_messages + new_messages, None


def create_error_command(
    updated_messages: List[BaseMessage],
    error_message: AIMessage,
    step_count: int,
    additional_updates: Optional[Dict[str, Any]] = None,
) -> Command:
    """Create a Command to END with error information.

    Args:
        updated_messages: Updated chat messages
        error_message: Error message to return
        step_count: Current step count
        additional_updates: Optional additional state updates

    Returns:
        Command routing to END with error state
    """
    updates = {
        "chat_messages": updated_messages,
        "script": None,
        "final_answer": error_message.content,
        "execution_complete": True,
        "error": error_message.content,
        "step_count": step_count + 1,
    }
    if additional_updates:
        updates.update(additional_updates)

    return Command(goto=END, update=updates)


async def create_find_tools_tool(
    all_tools: Sequence[StructuredTool],
    all_apps: List[Any],
) -> StructuredTool:
    """Create a find_tools StructuredTool for tool discovery.

    Args:
        all_tools: All available tools to search through
        all_apps: All available app definitions

    Returns:
        StructuredTool configured for finding relevant tools
    """

    async def find_tools_func(query: str):
        """Search for relevant tools from the connected applications based on a natural language query.

        Args:
            query: Natural language description of what you want to accomplish

        Returns:
            Top 4 matching tools with their details
        """
        return await PromptUtils.find_tools(query=query, all_tools=all_tools, all_apps=all_apps)

    return StructuredTool.from_function(
        func=find_tools_func,
        name="find_tools",
        description="Search for relevant tools from the connected applications based on a natural language query. Use this when you need to discover what tools are available for a specific task.",
    )


def create_cuga_lite_graph(
    model: BaseChatModel,
    prompt: Optional[str] = None,
    tool_provider: ToolProviderInterface = None,
    apps_list: Optional[List[str]] = None,
    agent_state: Optional[AgentState] = None,
    thread_id: Optional[str] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
) -> StateGraph:
    """
    Create a unified CugaLite subgraph combining CodeAct and CugaAgent functionality.

    Args:
        model: The language model to use
        prompt: Optional static prompt (if None, will be created dynamically from state)
        tool_provider: Tool provider interface for accessing tools
        apps_list: List of app names for tool context
        agent_state: Optional AgentState for variables management
        thread_id: Thread ID for E2B sandbox caching
        callbacks: Optional list of callback handlers

    Returns:
        StateGraph implementing the CugaLite architecture
    """

    # Load prompt template
    prompt_path = Path(__file__).parent / "prompts" / "mcp_prompt.jinja2"
    prompt_template = load_one_prompt(str(prompt_path), relative_to_caller=False)
    instructions = get_all_instructions_formatted()

    # Factory function to create prepare_tools_and_apps node with access to tools/config
    def create_prepare_node(base_tool_provider, base_prompt_template, base_instructions, tools_context_dict):
        """Factory to create prepare node with closure over tool provider and config."""

        async def prepare_tools_and_apps(state: CugaLiteState) -> Command:
            """Prepare tools, apps, and prompt once at the start of the graph.

            This node gets tools from tool_provider, filters based on state configuration,
            determines if find_tools should be enabled, and prepares the prompt.
            Tools are available via closure (per graph instance), prompt is stored in state.
            """
            if not base_tool_provider:
                raise ValueError("tool_provider is required")

            # Get tools from provider
            apps_for_prompt = None

            # Get apps from state and filter tools if specific app is selected
            if state.sub_task_app:
                # Specific app selected - filter tools to only this app
                all_apps = await base_tool_provider.get_apps()
                apps_for_prompt = [app for app in all_apps if app.name == state.sub_task_app]
                # Get only tools for this specific app
                tools_for_execution = await base_tool_provider.get_tools(state.sub_task_app)
                logger.info(f"Filtered to {len(tools_for_execution)} tools for app '{state.sub_task_app}'")
            elif state.api_intent_relevant_apps:
                # Filter to API apps
                all_apps = await base_tool_provider.get_apps()
                apps_for_prompt = [
                    app
                    for app in state.api_intent_relevant_apps
                    if hasattr(app, 'type') and app.type == 'api'
                ]
                # Get tools only for the identified apps
                tools_for_execution = []
                for app in apps_for_prompt:
                    app_tools = await base_tool_provider.get_tools(app.name)
                    tools_for_execution.extend(app_tools)
                logger.info(
                    f"Filtered to {len(tools_for_execution)} tools for {len(apps_for_prompt)} identified apps"
                )
            else:
                # Get all tools and apps
                all_apps = await base_tool_provider.get_apps()
                apps_for_prompt = all_apps
                tools_for_execution = await base_tool_provider.get_all_tools()

            # Determine enable_find_tools based on tool count
            tool_count = len(tools_for_execution) if tools_for_execution else 0
            threshold = settings.advanced_features.shortlisting_tool_threshold
            enable_find_tools = tool_count > threshold

            if enable_find_tools:
                logger.info(f"Auto-enabling find_tools: {tool_count} tools exceeds threshold of {threshold}")

            # Prepare prompt
            is_autonomous_subtask = bool(state.sub_task)
            # TODO: Add task loaded from file support this happens when we load file as playboook
            task_loaded_from_file = False  # Not used in current flow

            # Prepare tools for prompt - if find_tools enabled, only expose find_tools
            tools_for_prompt = tools_for_execution
            if enable_find_tools:
                find_tool = await create_find_tools_tool(
                    all_tools=tools_for_execution, all_apps=apps_for_prompt
                )
                tools_for_prompt = [find_tool]
                # Add find_tools to tools context for sandbox execution
                tools_context_dict['find_tools'] = find_tool.func
                logger.info(
                    "Exposing only find_tools in prompt (all tools + find_tools available in execution context)"
                )

            # Update tools context with all execution tools
            for tool in tools_for_execution:
                tools_context_dict[tool.name] = tool.func

            # Create prompt dynamically
            dynamic_prompt = prompt
            if not dynamic_prompt:
                dynamic_prompt = create_mcp_prompt(
                    tools_for_prompt,
                    allow_user_clarification=True,
                    return_to_user_cases=None,
                    instructions=base_instructions,
                    apps=apps_for_prompt,
                    task_loaded_from_file=task_loaded_from_file,
                    is_autonomous_subtask=is_autonomous_subtask,
                    prompt_template=base_prompt_template,
                    enable_find_tools=enable_find_tools,
                )

            return Command(
                goto="call_model",
                update={"tools_prepared": True, "prepared_prompt": dynamic_prompt, "step_count": 0},
            )

        return prepare_tools_and_apps

    # Factory function to create call_model node with access to model
    def create_call_model_node(base_model, base_callbacks):
        """Factory to create call_model node with closure over model."""

        async def call_model(state: CugaLiteState, config: Optional[RunnableConfig] = None) -> Command:
            """Call the LLM to generate code or text response."""
            # Get prompt from state (tools are available via sandbox context, not needed here)
            dynamic_prompt = state.prepared_prompt

            # Convert BaseMessage objects to dict format for model invocation
            messages_for_model = [{"role": "system", "content": dynamic_prompt}]

            # Check if we have variables and this is a new question (not a follow-up with existing AI responses)
            # If this is a new question (1 user msg, 0 AI msgs) or follow-up, add variables to the last user message
            var_manager = state.variables_manager
            existing_variable_names = var_manager.get_variable_names()
            variables_summary_text = None

            if existing_variable_names and state.sub_task_app:
                variables_summary_text = var_manager.get_variables_summary(
                    variable_names=existing_variable_names
                )
                variables_addendum = f"\n\n## Available Variables\n\n{variables_summary_text}\n\nYou can use these variables directly by their names."
                logger.info(
                    f"Will add variables summary for {len(existing_variable_names)} variables to user message"
                )

            logger.info(f"Processing {len(state.chat_messages)} chat messages for model invocation")
            for i, msg in enumerate(state.chat_messages):
                msg_type = type(msg).__name__
                msg_role = getattr(msg, 'type', None)
                logger.debug(
                    f"Message {i}: type={msg_type}, role={msg_role}, isinstance(HumanMessage)={isinstance(msg, HumanMessage)}, isinstance(AIMessage)={isinstance(msg, AIMessage)}"
                )

                if isinstance(msg, HumanMessage):
                    content = msg.content
                    # Add variables summary to the LAST user message only
                    if variables_summary_text and i == len(state.chat_messages) - 1:
                        content = content + variables_addendum
                        logger.debug("Added variables summary to last user message")
                    messages_for_model.append({"role": "user", "content": content})
                elif isinstance(msg, AIMessage):
                    messages_for_model.append({"role": "assistant", "content": msg.content})
                else:
                    # Handle generic BaseMessage by checking the 'type' attribute
                    if msg_role == 'human' or msg_role == 'user':
                        content = msg.content
                        if variables_summary_text and i == len(state.chat_messages) - 1:
                            content = content + variables_addendum
                        messages_for_model.append({"role": "user", "content": content})
                        logger.debug(f"Added BaseMessage as user message (role={msg_role})")
                    elif msg_role == 'ai' or msg_role == 'assistant':
                        messages_for_model.append({"role": "assistant", "content": msg.content})
                        logger.debug(f"Added BaseMessage as assistant message (role={msg_role})")
                    else:
                        logger.warning(
                            f"Skipping message {i} with unknown type: {msg_type}, role: {msg_role}"
                        )

            logger.debug(f"Total messages for model (including system): {len(messages_for_model)}")

            # Get configurable values from config
            configurable = config.get("configurable", {}) if config else {}
            current_callbacks = configurable.get("callbacks", base_callbacks or [])

            # Invoke model with callbacks
            response = await base_model.ainvoke(messages_for_model, config={"callbacks": current_callbacks})

            content = response.content
            reasoning_content = response.additional_kwargs.get('reasoning_content')

            tracker.collect_step(step=Step(name="Raw_Assistant_Response", data=content))

            if not content or (reasoning_content and '```python' in reasoning_content):
                content = reasoning_content or content

            code = extract_and_combine_codeblocks(content)

            if code:
                tracker.collect_step(step=Step(name="Assistant_code", data=content))
                logger.debug(
                    f"\n{'=' * 50} ASSISTANT CODE {'=' * 50}\n{code}\n{'=' * 50} END ASSISTANT CODE {'=' * 50}"
                )
                updated_messages, error_message = append_chat_messages_with_step_limit(
                    state, [AIMessage(content=content)]
                )
                if error_message:
                    return create_error_command(updated_messages, error_message, state.step_count)

                return Command(
                    goto="sandbox",
                    update={
                        "chat_messages": updated_messages,
                        "script": code,
                        "step_count": state.step_count + 1,
                    },
                )
            else:
                tracker.collect_step(step=Step(name="Assistant_nl", data=content))
                planning_response = response.content

                updated_messages, error_message = append_chat_messages_with_step_limit(
                    state, [AIMessage(content=planning_response)]
                )
                if error_message:
                    return create_error_command(updated_messages, error_message, state.step_count)

                return Command(
                    goto=END,
                    update={
                        "chat_messages": updated_messages,
                        "script": None,
                        "final_answer": planning_response,
                        "execution_complete": True,
                        "step_count": state.step_count + 1,
                    },
                )

        return call_model

    # Factory function to create sandbox node with access to tools context
    def create_sandbox_node(base_tools_context, base_thread_id, base_apps_list):
        """Factory to create sandbox node with closure over tools context and config."""

        async def sandbox(state: CugaLiteState, config: Optional[RunnableConfig] = None):
            """Execute code in sandbox and return results."""
            # Get configurable values from config
            configurable = config.get("configurable", {}) if config else {}
            current_thread_id = configurable.get("thread_id", base_thread_id)
            current_apps_list = configurable.get("apps_list", base_apps_list)

            # Get existing variables using CugaLiteState's own variables_manager
            existing_vars = {}
            for var_name in state.variables_manager.get_variable_names():
                existing_vars[var_name] = state.variables_manager.get_variable(var_name)

            # Add tools to context
            context = {**existing_vars, **base_tools_context}

            try:
                # Execute the script - pass the CugaLiteState itself since it has variables_manager
                output, new_vars = await CodeExecutor.eval_with_tools_async(
                    code=state.script,
                    _locals=context,
                    state=state,  # Pass CugaLiteState - it has variables_manager property
                    thread_id=current_thread_id,
                    apps_list=current_apps_list,
                )

                tracker.collect_step(step=Step(name="User_output", data=output))
                tracker.collect_step(step=Step(name="User_output_variables", data=json.dumps(new_vars)))

                logger.debug(
                    f"\n\n------\n\n📝 Execution output:\n\n {output.strip()[:2000]}{'...' if len(output.strip()) > 2000 else ''} \n\n------\n\n"
                )

                # Update variables using CugaLiteState's variables_manager
                # This automatically updates state.variables_storage
                for name, value in new_vars.items():
                    state.variables_manager.add_variable(
                        value, name=name, description="Created during code execution"
                    )

                tracker.collect_step(
                    step=Step(
                        name="User_return",
                        data=f"Execution output preview:\n{output.strip()[:2500]}{'...' if len(output.strip()) > 2500 else ''} Execution output:\n{output}",
                    )
                )

                new_message = HumanMessage(
                    content=f"Execution output preview:\n{output.strip()[:2500]}{'...' if len(output.strip()) > 2500 else ''} Execution output:\n{output}"
                )
                updated_messages, error_message = append_chat_messages_with_step_limit(state, [new_message])

                if error_message:
                    return create_error_command(
                        updated_messages,
                        error_message,
                        state.step_count,
                        additional_updates={
                            "variables_storage": state.variables_storage,
                            "variable_counter_state": state.variable_counter_state,
                            "variable_creation_order": state.variable_creation_order,
                        },
                    )

                return {
                    "chat_messages": updated_messages,
                    "variables_storage": state.variables_storage,
                    "variable_counter_state": state.variable_counter_state,
                    "variable_creation_order": state.variable_creation_order,
                    "step_count": state.step_count + 1,
                }
            except Exception as e:
                error_msg = f"Error during execution: {str(e)}"
                logger.error(error_msg)
                new_message = HumanMessage(content=error_msg)
                updated_messages, limit_error_message = append_chat_messages_with_step_limit(
                    state, [new_message]
                )

                if limit_error_message:
                    return create_error_command(updated_messages, limit_error_message, state.step_count)

                return {
                    "chat_messages": updated_messages,
                    "error": error_msg,
                    "execution_complete": True,
                    "step_count": state.step_count + 1,
                }

        return sandbox

    # Create mutable tools context that will be populated by prepare_node
    tools_context = {}

    # Create node instances using factories
    prepare_node = create_prepare_node(tool_provider, prompt_template, instructions, tools_context)
    call_model_node = create_call_model_node(model, callbacks)
    sandbox_node = create_sandbox_node(tools_context, thread_id, apps_list)

    # Build the graph
    graph = StateGraph(CugaLiteState)
    graph.add_node("prepare_tools_and_apps", prepare_node)
    graph.add_node("call_model", call_model_node)
    graph.add_node("sandbox", sandbox_node)

    graph.add_edge(START, "prepare_tools_and_apps")
    graph.add_edge("sandbox", "call_model")

    return graph
