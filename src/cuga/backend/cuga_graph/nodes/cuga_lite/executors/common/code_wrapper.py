class CodeWrapper:
    """Handles wrapping user code for async execution."""

    @staticmethod
    def wrap_code(code: str) -> str:
        """Wrap user code in an async function for execution.

        Args:
            code: User's Python code

        Returns:
            Wrapped code ready for execution
        """
        indented_code = '\n'.join('    ' + line for line in code.split('\n'))
        lines = [line.strip() for line in code.split('\n') if line.strip()]

        if lines and not lines[-1].startswith(('print', 'return', '#')) and '=' not in lines[-1]:
            indented_code += f"\n    print({lines[-1]})"

        wrapped_code = f"""
import asyncio
async def _async_main():
{indented_code}
    return locals()

# Execute the wrapped function
"""
        return wrapped_code
