"""
Secure API key management for Blueprint Processor.
Loads credentials from environment variables or .env file.
"""

import os
from pathlib import Path
from functools import lru_cache


def load_env_file(env_path: Path = None) -> None:
    """Load environment variables from .env file."""
    if env_path is None:
        env_path = Path(__file__).parent.parent / ".env"

    if not env_path.exists():
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key not in os.environ:  # Don't override existing env vars
                    os.environ[key] = value


@lru_cache(maxsize=1)
def get_api_key(key_name: str = "ANTHROPIC_API_KEY") -> str:
    """
    Get API key from environment variable.

    Priority:
    1. Environment variable (set in shell or system)
    2. .env file in project root

    Raises:
        ValueError: If API key is not found
    """
    load_env_file()

    api_key = os.environ.get(key_name)

    if not api_key:
        raise ValueError(
            f"{key_name} not found. Set it via:\n"
            f"  1. Environment variable: export {key_name}=your-key\n"
            f"  2. Create .env file with: {key_name}=your-key"
        )

    return api_key


# Convenience function
def get_anthropic_key() -> str:
    """Get Anthropic API key."""
    return get_api_key("ANTHROPIC_API_KEY")
