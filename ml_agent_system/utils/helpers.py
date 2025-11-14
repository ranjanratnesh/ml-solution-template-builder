"""Helper utility functions."""

import json
from typing import Any, Dict
from datetime import datetime


def save_json(data: Dict[str, Any], filepath: str):
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary loaded from file
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def timestamp() -> str:
    """
    Get current timestamp as string.

    Returns:
        ISO format timestamp
    """
    return datetime.now().isoformat()


def print_section(title: str, width: int = 80):
    """
    Print a formatted section header.

    Args:
        title: Section title
        width: Width of the header
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")
