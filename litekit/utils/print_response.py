"""Output formatting utilities for MedKit CLI modules.

This module provides unified output formatting functions for displaying
results in a consistent, readable format.
"""

from typing import Any, Dict, Union
from pydantic import BaseModel


def print_response(
    result: Union[BaseModel, Dict[str, Any], Any],
    title: str = "Result"
) -> None:
    """Print result in a formatted manner.

    Args:
        result: The result to print (BaseModel, dict, or other)
        title: Title for the output
    """
    print(f"\n=== {title.upper()} ===")

    if result is None:
        print("No result to display")
        return

    # Convert Pydantic models to dict
    if isinstance(result, BaseModel):
        result_dict = result.model_dump()
    elif isinstance(result, dict):
        result_dict = result
    else:
        print(str(result))
        return

    # Display dictionary results
    for section_name, section_value in result_dict.items():
        if section_value is not None:
            print(f"\n{section_name.replace('_', ' ').upper()}:")
            _format_value(section_value, indent=1)


def _format_value(value: Any, indent: int = 0) -> None:
    """Format and print a value.

    Args:
        value: The value to format
        indent: Indentation level
    """
    indent_str = "  " * indent

    if isinstance(value, dict):
        for k, v in value.items():
            key_display = k.replace('_', ' ').title()
            if isinstance(v, (dict, list)):
                print(f"{indent_str}{key_display}:")
                _format_value(v, indent + 1)
            else:
                print(f"{indent_str}{key_display}: {v}")

    elif isinstance(value, list):
        if not value:
            print(f"{indent_str}(empty)")
            return

        for i, item in enumerate(value, 1):
            if isinstance(item, (dict, list)):
                print(f"{indent_str}{i}. ")
                _format_value(item, indent + 1)
            else:
                print(f"{indent_str}{i}. {item}")

    else:
        print(f"{indent_str}{value}")


def print_simple_result(
    result: Union[str, Dict[str, Any], None],
    title: str = "Result"
) -> None:
    """Print a simple result (string or flat dict).

    Args:
        result: String or dictionary to display
        title: Title for the output
    """
    print(f"\n=== {title.upper()} ===")

    if result is None:
        print("No result to display")
        return

    if isinstance(result, dict):
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        print(str(result))
