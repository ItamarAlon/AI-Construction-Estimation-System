from typing import TypedDict, Optional
from pathlib import Path

class Skill(TypedDict):
    """A skill that can be progressively disclosed to the agent."""
    name: str  # Unique identifier for the skill
    description: str  # 1-2 sentence description to show in system prompt
    content: Optional[str]  # Full skill content with detailed instructions
    path: Path