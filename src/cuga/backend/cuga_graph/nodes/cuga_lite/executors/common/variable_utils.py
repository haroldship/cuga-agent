import types
from typing import Any, Set
from loguru import logger


class VariableUtils:
    """Utilities for managing variables during code execution."""

    @staticmethod
    def is_serializable(value: Any) -> bool:
        """Check if a value is serializable.

        Args:
            value: Value to check

        Returns:
            True if value is serializable, False otherwise
        """
        if isinstance(value, (str, int, float, bool, type(None))):
            return True

        if isinstance(value, (list, tuple)):
            return all(VariableUtils.is_serializable(item) for item in value)

        if isinstance(value, dict):
            return all(
                VariableUtils.is_serializable(k) and VariableUtils.is_serializable(v)
                for k, v in value.items()
            )

        if isinstance(
            value, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType, type)
        ):
            return False

        try:
            import pandas as pd

            if isinstance(value, (pd.DataFrame, pd.Series)):
                return True
        except ImportError:
            pass

        return False

    @staticmethod
    def filter_new_variables(all_locals: dict[str, Any], original_keys: Set[str]) -> dict[str, Any]:
        """Filter and return only new, serializable variables.

        Args:
            all_locals: Dictionary of all local variables
            original_keys: Set of keys that existed before execution

        Returns:
            Dictionary of new serializable variables (preserves insertion order)
        """
        new_keys = set(all_locals.keys()) - original_keys
        new_vars = {}

        for key in all_locals.keys():
            if key not in new_keys:
                continue
            if key.startswith('_'):
                continue

            value = all_locals[key]
            if VariableUtils.is_serializable(value):
                new_vars[key] = value
            else:
                logger.debug(f"Skipping non-serializable variable '{key}': {type(value).__name__}")

        return new_vars

    @staticmethod
    def reorder_variables_by_print(new_vars: dict[str, Any], code: str) -> dict[str, Any]:
        """Reorder variables to move printed ones to the end.

        Args:
            new_vars: Dictionary of new variables
            code: Original code to analyze

        Returns:
            Reordered dictionary with printed variables at the end
        """
        if not new_vars:
            return new_vars

        lines = code.strip().split('\n')
        last_print_line = None

        for line in reversed(lines):
            stripped = line.strip()
            if 'print(' in stripped:
                last_print_line = stripped
                break

        if not last_print_line:
            return new_vars

        reordered_vars = {}
        print_vars = {}

        for var_name, var_value in new_vars.items():
            if var_name in last_print_line and len(var_name) > 3:
                print_vars[var_name] = var_value
            else:
                reordered_vars[var_name] = var_value

        reordered_vars.update(print_vars)
        return reordered_vars

    @staticmethod
    def add_variables_to_manager(new_vars: dict[str, Any], var_manager, result: str) -> str:
        """Add new variables to VariablesManager and append summary to result.

        Args:
            new_vars: Dictionary of new variables
            var_manager: VariablesManager instance
            result: Current execution result string

        Returns:
            Updated result string with variables summary
        """
        if not new_vars:
            return result

        for var_name, var_value in new_vars.items():
            var_manager.add_variable(var_value, name=var_name, description="Created during code execution")

        try:
            variables_summary = var_manager.get_variables_summary(variable_names=list(new_vars.keys()))
            if variables_summary and variables_summary != "# No variables stored":
                result += f"\n\n## New Variables Created:\n{variables_summary}"
        except Exception as e:
            logger.debug(f"Could not generate variables summary: {e}")

        return result
