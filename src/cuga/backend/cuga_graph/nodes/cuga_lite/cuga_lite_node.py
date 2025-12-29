"""
CugaLite Node - Fast execution node using CugaLite subgraph
"""

import json
from typing import Literal, Dict, Any, List, Optional
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field

from cuga.backend.cuga_graph.nodes.shared.base_node import BaseNode
from cuga.backend.cuga_graph.state.agent_state import AgentState, SubTaskHistory
from cuga.backend.activity_tracker.tracker import ActivityTracker
from cuga.backend.cuga_graph.nodes.api.api_planner_agent.prompts.load_prompt import ActionName
from cuga.backend.cuga_graph.state.api_planner_history import CoderAgentHistoricalOutput
from langchain_core.messages import HumanMessage
from cuga.backend.llm.utils.helpers import load_one_prompt
from cuga.config import settings


tracker = ActivityTracker()


def _convert_sets_to_lists(value: Any) -> Any:
    """Recursively convert sets to lists for JSON serialization."""
    if isinstance(value, set):
        return list(value)
    elif isinstance(value, dict):
        return {k: _convert_sets_to_lists(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_convert_sets_to_lists(item) for item in value]
    else:
        return value


class CugaLiteOutput(BaseModel):
    """Output model for CugaLite execution (similar to CodeAgentOutput)."""

    code: str = ""
    execution_output: str = ""
    steps_summary: List[str] = Field(default_factory=list)
    summary: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)
    final_answer: str = ""


class CugaLiteNode(BaseNode):
    """Node wrapper for routing to CugaLite subgraph."""

    def __init__(self, langfuse_handler: Optional[Any] = None, prompt_template: Optional[str] = None):
        super().__init__()
        self.name = "CugaLite"
        self.prompt_template = load_one_prompt('prompts/mcp_prompt.jinja2')
        self.langfuse_handler = langfuse_handler

    @staticmethod
    async def read_text_file(file_path: str) -> Optional[str]:
        """Read text file content using filesystem tool via registry.

        Args:
            file_path: Path to the file to read

        Returns:
            File content as string, or None if failed
        """
        try:
            from cuga.backend.cuga_graph.nodes.cuga_lite.tool_registry_provider import call_api

            result = await call_api(
                app_name="filesystem", api_name="filesystem_read_text_file", args={"path": file_path}
            )

            if isinstance(result, dict):
                if "error" in result:
                    logger.error(f"Error reading file {file_path}: {result['error']}")
                    return None
                if "result" in result:
                    return result["result"]
                if "content" in result:
                    return result["content"]
                return str(result)
            elif isinstance(result, str):
                return result
            else:
                logger.error(f"Unexpected result type from read_text_file: {type(result)}")
                return None

        except Exception as e:
            logger.error(f"Exception reading file {file_path}: {e}")
            return None

    @staticmethod
    def _get_new_variable_names(state: AgentState, initial_var_names: List[str]) -> List[str]:
        """Get names of variables created during execution in chronological order.

        Args:
            state: Current agent state
            initial_var_names: Variable names before execution

        Returns:
            List of new variable names in chronological creation order
        """
        # Use creation_order to ensure chronological ordering
        current_var_names = state.variable_creation_order
        return [name for name in current_var_names if name not in initial_var_names]

    @staticmethod
    def _log_variable_changes(state: AgentState, initial_var_names: List[str]) -> None:
        """Log variable changes after execution.

        Args:
            state: Current agent state
            initial_var_names: Variable names before execution
        """
        current_var_names = state.variables_manager.get_variable_names()
        new_var_names = CugaLiteNode._get_new_variable_names(state, initial_var_names)
        logger.info(
            f"Variables before: {len(initial_var_names)}, "
            f"after: {len(current_var_names)}, "
            f"new: {len(new_var_names)}"
        )

    def _generate_fallback_answer(self, state: AgentState, new_var_names: List[str]) -> str:
        """Generate fallback answer when execution result is empty.

        Args:
            state: Current agent state
            new_var_names: List of newly created variable names

        Returns:
            Fallback answer string
        """
        if not new_var_names:
            return "Task completed successfully."

        last_var_name = new_var_names[-1]
        last_var_value = state.variables_manager.get_variable(last_var_name)

        if last_var_value is None:
            return "Task completed successfully."

        if isinstance(last_var_value, (list, dict)):
            try:
                answer = json.dumps(last_var_value, indent=2, default=str)
            except Exception:
                answer = str(last_var_value)
        else:
            answer = str(last_var_value)

        logger.info(f"Using last variable '{last_var_name}' value as fallback answer")
        return answer

    @staticmethod
    def _has_error(answer: str) -> bool:
        """Check if answer contains error indicators.

        Args:
            answer: Answer string to check

        Returns:
            True if error indicators found, False otherwise
        """
        if not answer:
            return False

        error_indicators = ['Error during execution:', 'Error:', 'Exception:', 'Traceback', 'Failed to']
        return any(indicator in answer for indicator in error_indicators)

    async def node(self, state: AgentState) -> Command[Literal['FinalAnswerAgent', 'PlanControllerAgent']]:
        """Execute CugaLite graph wrapper and process results.

        Args:
            state: Current agent state

        Returns:
            Command to route to FinalAnswerAgent or PlanControllerAgent
        """
        logger.info(f"CugaLite executing - state.input: {state.input}")
        logger.info(f"CugaLite executing - state.sub_task: {state.sub_task}")

        # Add initialization message
        # state.messages.append(
        #     AIMessage(
        #         content=json.dumps(
        #             {
        #                 "status": "initializing",
        #                 "message": f"Initializing CugaLite with {len(state.api_intent_relevant_apps) if state.api_intent_relevant_apps else 'all'} apps",
        #             }
        #         )
        #     )
        # )

        # Use sub_task as the input if available (preferred over state.input)
        task_input = state.sub_task if state.sub_task else state.input
        is_autonomous_subtask = bool(state.sub_task)
        logger.info(f"Using task_input: {task_input}")
        logger.info(f"is_autonomous_subtask: {is_autonomous_subtask}")

        # Check if task_input is just a markdown file path and replace with file content
        task_input_stripped = task_input.strip()
        if (
            task_input_stripped.endswith('.md')
            and '\n' not in task_input_stripped
            and ' ' not in task_input_stripped
        ):
            logger.info(f"Detected markdown file path: {task_input_stripped}")
            try:
                file_content = await self.read_text_file(task_input_stripped)
                if file_content:
                    task_input = file_content
                    task_input += "\n\nDo not use cuga_kowledge.md for the above task."
                    logger.info(f"Replaced task input with file content from {task_input_stripped}")
                else:
                    logger.warning(f"Failed to read file {task_input_stripped}, using original task input")
            except Exception as e:
                logger.warning(
                    f"Error reading markdown file {task_input_stripped}: {e}, using original task input"
                )

        # Determine app configuration
        app_names = None
        if state.sub_task_app:
            app_names = [state.sub_task_app]
            logger.info(f"Using app from state.sub_task_app: {app_names}")
        elif state.api_intent_relevant_apps:
            app_names = [app.name for app in state.api_intent_relevant_apps if app.type == 'api']
            logger.info(f"Using apps from state.api_intent_relevant_apps: {app_names}")

        # Store initial state in metadata for callback tracking
        initial_var_names = state.variables_manager.get_variable_names()
        logger.info(f"Storing {len(initial_var_names)} initial variable names for tracking")

        # Add the user's input to chat_messages (shared key)
        # Preserve existing chat history for multi-turn conversations
        updated_chat_messages = list(state.chat_messages) if state.chat_messages else []
        logger.info(f"Existing chat_messages count: {len(updated_chat_messages)}")
        if updated_chat_messages:
            logger.debug(f"Previous messages: {[type(msg).__name__ for msg in updated_chat_messages]}")

        updated_chat_messages.append(HumanMessage(content=task_input))
        logger.info(f"Added user input to chat_messages: {task_input[:100]}...")
        logger.info(f"Total chat_messages count after append: {len(updated_chat_messages)}")

        # Route to CugaLite subgraph with updated state
        logger.info("Routing to CugaLiteSubgraph")
        return Command(
            goto="CugaLiteSubgraph",
            update={
                "chat_messages": updated_chat_messages,
                "variables_storage": state.variables_storage,
                "variable_counter_state": state.variable_counter_state,
                "variable_creation_order": state.variable_creation_order,
                "sub_task": state.sub_task,
                "sub_task_app": state.sub_task_app,
                "api_intent_relevant_apps": state.api_intent_relevant_apps,
                "cuga_lite_metadata": {
                    "initial_var_names": initial_var_names,
                    "is_autonomous_subtask": is_autonomous_subtask,
                },
            },
        )

    async def callback_node(
        self, state: AgentState
    ) -> Command[Literal['FinalAnswerAgent', 'PlanControllerAgent']]:
        """Process results after CugaLite subgraph execution."""
        logger.info("CugaLite callback node - processing subgraph results")

        # Get metadata from state
        metadata = state.cuga_lite_metadata or {}
        initial_var_names = metadata.get("initial_var_names", [])
        is_autonomous_subtask = metadata.get("is_autonomous_subtask", False)

        answer = state.final_answer or "No answer found"
        logger.info(f"Extracted answer: {answer[:200]}...")

        # Log variable changes
        self._log_variable_changes(state, initial_var_names)

        # Process the results using the existing logic
        return await self._process_results(
            state=state,
            answer=answer,
            initial_var_names=initial_var_names,
            is_autonomous_subtask=is_autonomous_subtask,
        )

    async def _process_results(
        self,
        state: AgentState,
        answer: str,
        initial_var_names: List[str],
        is_autonomous_subtask: bool,
    ) -> Command[Literal['FinalAnswerAgent', 'PlanControllerAgent']]:
        """Process results from CugaLite graph execution and route to appropriate next node.

        Args:
            state: Agent state
            answer: Final answer from graph execution
            initial_var_names: Variable names before execution
            is_autonomous_subtask: Whether this is a subtask

        Returns:
            Command to route to FinalAnswerAgent or PlanControllerAgent
        """
        logger.info("Processing CugaLite execution results")
        logger.info(f"Answer: {answer[:200] if answer else 'None'}...")

        # Check for errors
        has_error = self._has_error(answer)
        if has_error:
            logger.warning(f"Detected error in answer content: {answer[:200]}...")

        if has_error:
            logger.error("CugaLite execution failed with error")
            logger.error(f"Full answer: {answer}")

            # Update state with error information
            if is_autonomous_subtask:
                # For sub-tasks, add error to history and return to plan controller

                if state.api_planner_history:
                    state.api_planner_history[-1].agent_output = CoderAgentHistoricalOutput(
                        variables_summary="Execution failed",
                        final_output=answer,
                    )

                state.stm_all_history.append(
                    SubTaskHistory(
                        sub_task=state.format_subtask(),
                        steps=[],
                        final_answer=answer,
                    )
                )
                state.last_planner_answer = answer
                state.sender = self.name
                logger.info("CugaLite sub-task execution failed, returning error to PlanControllerAgent")
                return Command(update=state.model_dump(), goto="PlanControllerAgent")
            else:
                # For regular execution, set final answer with error
                state.final_answer = answer
                state.sender = self.name
                logger.info("CugaLite execution failed, proceeding to FinalAnswerAgent with error")
                return Command(update=state.model_dump(), goto="FinalAnswerAgent")

        # chat_messages should already be synced since it's a shared key
        # But ensure they're properly formatted as BaseMessage objects
        if state.chat_messages:
            logger.info(f"Chat messages synced from subgraph: {len(state.chat_messages)} messages")

        # Variables are already synced via variables_storage (shared key)
        new_var_names = self._get_new_variable_names(state, initial_var_names)
        logger.info(
            f"After execution, variables_manager has {state.variables_manager.get_variable_count()} variables ({len(new_var_names)} new)"
        )

        # Check if answer is empty and provide a fallback
        if not answer or not answer.strip():
            logger.warning("Empty final answer detected, using fallback")
            answer = self._generate_fallback_answer(state, new_var_names)

        # Check if we're executing a sub-task
        if is_autonomous_subtask:
            # Sub-task execution - return to PlanControllerAgent

            # Keep only last N generated variables
            keep_last_n = settings.advanced_features.sub_task_keep_last_n
            if len(new_var_names) > keep_last_n:
                original_count = len(new_var_names)
                vars_to_keep = new_var_names[-keep_last_n:]
                vars_to_remove = new_var_names[:-keep_last_n]
                for var_name in vars_to_remove:
                    if state.variables_manager.remove_variable(var_name):
                        logger.debug(
                            f"Removed variable '{var_name}' to keep only last {keep_last_n} generated variables"
                        )
                new_var_names = vars_to_keep
                logger.info(f"Kept only last {keep_last_n} of {original_count} generated variables")

            state.api_last_step = ActionName.CONCLUDE_TASK
            state.guidance = None

            # Update api_planner_history with CoderAgentHistoricalOutput
            if state.api_planner_history:
                state.api_planner_history[-1].agent_output = CoderAgentHistoricalOutput(
                    variables_summary=state.variables_manager.get_variables_summary(
                        new_var_names, max_length=5000
                    )
                    if new_var_names
                    else "No new variables",
                    final_output=answer,
                )

            state.stm_all_history.append(
                SubTaskHistory(
                    sub_task=state.format_subtask(),
                    steps=[],
                    final_answer=answer,
                )
            )
            state.last_planner_answer = answer
            state.sender = self.name

            logger.info("CugaLite sub-task execution successful, proceeding to PlanControllerAgent")
            return Command(update=state.model_dump(), goto="PlanControllerAgent")
        else:
            # Regular execution - proceed to FinalAnswerAgent
            state.final_answer = answer
            state.sender = self.name
            logger.info("CugaLite execution successful, proceeding to FinalAnswerAgent")
            return Command(update=state.model_dump(), goto="FinalAnswerAgent")
