from langchain.tools import tool

DESCRIPTION = """
[description of the tool]
"""

@tool(description=DESCRIPTION) #optional description
def some_tool(input):
    pass # replace with the actual tool logic

@tool
def get_weather(location: str) -> str:
    """Get weather for location."""
    return f"It's sunny in {location}"
