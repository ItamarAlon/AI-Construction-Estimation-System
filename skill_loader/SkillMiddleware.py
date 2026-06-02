from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import SystemMessage, ToolMessage
from langchain.tools import tool
from typing import Callable

try:
    from .SkillRegistry import SkillRegistry
except ImportError:
    from SkillRegistry import SkillRegistry

class SkillMiddleware(AgentMiddleware):
    """Middleware that injects skill descriptions into the system prompt."""

    _registry: SkillRegistry

    @tool
    def load_skill(skill_name: str) -> str:
        """Load the full content of a skill into the agent's context.

        Use this when you need detailed information about how to handle a specific
        type of request. This will provide you with comprehensive instructions,
        policies, and guidelines for the skill area.

        Returns:
            The full content of the skill.

        Args:
            skill_name: The name of the skill to load (e.g., "expense_reporting", "travel_booking")
        """
        skill = SkillMiddleware._registry.get_skill_by_name(skill_name)
        print(f"Loaded skill")
        return f"Loaded skill: {skill_name}\n\n{SkillMiddleware._registry.get_skill_content(skill)}"

    tools = [load_skill]

    def __init__(self, *skills_folders: str | None):
        SkillMiddleware._registry = SkillRegistry(*skills_folders)
        self._loaded_skills: set[str] = set()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync: Inject skill descriptions into system prompt."""
        available_skills = [
            skill for skill in self._registry.get_all_skills() if skill.get("name") not in self._loaded_skills
        ]
        if available_skills:
            skills_prompt = "\n".join(
                f"- **{skill.get('name')}**: {skill.get('description')}"
                for skill in available_skills
            )
        else:
            skills_prompt = "No unloaded skills remain for this run."

        skills_addendum = (
            f"\n\n## Available Skills\n\n{skills_prompt}\n\n"
            "Use the load_skill tool only when needed for missing details. "
            "Never call load_skill twice for the same skill in the same run."
        )

        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        tools = request.tools
        if not available_skills:
            tools = [tool for tool in request.tools if getattr(tool, "name", None) != "load_skill"]

        modified_request = request.override(
            system_message=new_system_message,
            tools=tools,
        )
        return handler(modified_request)

    def before_agent(self, state, runtime):
        """Reset skill tracking for each agent invocation."""
        self._loaded_skills.clear()
        return None

    def wrap_tool_call(self, request, handler):
        """Skip duplicate load_skill calls for the same skill."""
        tool_name = request.tool.name if request.tool else request.tool_call.get("name")
        if tool_name != "load_skill":
            return handler(request)

        args = request.tool_call.get("args") or {}
        skill_name = args.get("skill_name")
        if not skill_name:
            return handler(request)

        if skill_name in self._loaded_skills:
            remaining_skills = [
                skill.get("name")
                for skill in self._registry.get_all_skills()
                if skill.get("name") not in self._loaded_skills
            ]
            remaining_hint = (
                f"Remaining skills: {', '.join(remaining_skills)}."
                if remaining_skills
                else "No other skills remain; answer using already loaded content."
            )
            return ToolMessage(
                content=(
                    f"Skill '{skill_name}' is already loaded. "
                    "Do not call it again. "
                    f"{remaining_hint}"
                ),
                tool_call_id=request.tool_call["id"],
                name="load_skill",
                status="error",
            )

        self._loaded_skills.add(skill_name)
        return handler(request)
