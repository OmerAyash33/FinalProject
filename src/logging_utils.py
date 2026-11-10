"""Tiny shared helper so every stage logs with the same `[HH:MM:SS] message` format."""
from datetime import datetime


def log(message: str) -> None:
    """Print `message` prefixed with the current wall-clock time (HH:MM:SS)."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
