import uuid
from langchain_core.tools import tool
from langgraph.types import Command
from AgentBuilder import AgentBuilder

USE_SHORTHAND = False  # set to True to use ask_user_when_needed(), False for the explicit version

# Pattern: agent asks for missing information before activating a tool
#
# 1. Define a placeholder ask_user tool — its implementation doesn't matter
#    because it will never actually execute.
# 2. Configure HITL with interrupt_on={"ask_user": True}
# 3. When the agent decides it needs info, it calls ask_user(question="...") —
#    the interrupt fires.
# 4. Resume with type: "respond" and your message is injected directly as the
#    tool result, skipping the tool body entirely.

@tool
def ask_user(question: str) -> str:
    """Ask the user for missing information before proceeding."""
    return ""  # never runs — HITL intercepts it

@tool
def write_file(filename: str, content: str) -> str:
    """Write content to a file."""
    with open(filename, "w") as f:
        f.write(content)
    return f"Wrote to {filename}"

@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query."""
    return f"Executed: {query}"

# --- Option 1: explicit version ---
# Manually define ask_user, pass it in, and configure interrupt_on yourself.
# Use this when you want full control over the tool definition or interrupt config.

# --- Option 2: shorthand version ---
# ask_user_when_needed() creates the ask_user tool internally and wires everything up.
# Use this when you just want the behavior without the boilerplate.

if USE_SHORTHAND:
    agent = AgentBuilder(
        model="gpt-4o",
        tools=[write_file, execute_sql],
        system_prompt="If you need information from the user before calling a tool, use ask_user first."
    ).ask_user_when_needed().build()
else:
    agent = AgentBuilder(
        model="gpt-4o",
        tools=[ask_user, write_file, execute_sql],
        system_prompt="If you need information from the user before calling a tool, use ask_user first."
    ).human_in_the_loop(
        interrupt_on={"ask_user": True}
    ).build()

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# First invoke — agent calls ask_user, triggers interrupt
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Save my report"}]},
    config=config,
    version="v2",
)
print(result.interrupts)  # shows the question the agent asked

# Resume — your answer becomes the tool result.
# The key difference from approve/reject is that "respond" skips the tool
# entirely — the message you provide is returned to the agent as if the tool
# returned it. The agent then has the missing info and continues to the real
# tool call.
agent.invoke(
    Command(resume={"decisions": [{"type": "respond", "message": "Save it as report_2026.pdf"}]}),
    config=config,
    version="v2",
)
