import asyncio
import json


class RestrictedEnvironment:
    """Manages restricted execution environment for local code execution."""

    @staticmethod
    def create_restricted_import(allowed_modules: set):
        """Create a restricted import function.

        Args:
            allowed_modules: Set of allowed module names

        Returns:
            Restricted import function
        """
        _original_import = (
            __builtins__['__import__'] if isinstance(__builtins__, dict) else __builtins__.__import__
        )

        def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split('.')[0] not in allowed_modules:
                raise ImportError(f"Import of '{name}' is not allowed in restricted execution context")
            return _original_import(name, globals, locals, fromlist, level)

        return restricted_import

    @staticmethod
    def create_safe_builtins(restricted_import_func) -> dict:
        """Create a dictionary of safe builtin functions.

        Args:
            restricted_import_func: The restricted import function to use

        Returns:
            Dictionary of safe builtins
        """
        return {
            'dict': dict,
            'list': list,
            'tuple': tuple,
            'set': set,
            'frozenset': frozenset,
            'str': str,
            'bytes': bytes,
            'bytearray': bytearray,
            'int': int,
            'float': float,
            'bool': bool,
            'complex': complex,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'any': any,
            'all': all,
            'chr': chr,
            'ord': ord,
            'format': format,
            'repr': repr,
            'isinstance': isinstance,
            'issubclass': issubclass,
            'type': type,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            'delattr': delattr,
            'iter': iter,
            'next': next,
            'slice': slice,
            'BaseException': BaseException,
            'Exception': Exception,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'KeyError': KeyError,
            'IndexError': IndexError,
            'AttributeError': AttributeError,
            'RuntimeError': RuntimeError,
            'StopIteration': StopIteration,
            'AssertionError': AssertionError,
            'ImportError': ImportError,
            'print': print,
            'None': None,
            'True': True,
            'False': False,
            'locals': locals,
            'vars': vars,
            '__name__': '__restricted__',
            '__build_class__': __build_class__,
            '__import__': restricted_import_func,
        }

    @staticmethod
    def create_restricted_globals(safe_builtins: dict, safe_locals: dict) -> dict:
        """Create restricted globals dictionary.

        Args:
            safe_builtins: Dictionary of safe builtin functions
            safe_locals: Dictionary of safe local variables/tools

        Returns:
            Dictionary of restricted globals
        """
        restricted_globals = {
            "__builtins__": safe_builtins,
            "asyncio": asyncio,
            "json": json,
        }

        try:
            import pandas as pd

            restricted_globals["pd"] = pd
            restricted_globals["pandas"] = pd
        except ImportError:
            pass

        restricted_globals.update(safe_locals)
        return restricted_globals
