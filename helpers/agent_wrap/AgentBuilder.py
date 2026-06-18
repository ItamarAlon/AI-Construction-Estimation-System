from Agent import Agent
import asyncio
import inspect
#from langchain.agents import create_agent
from skill_loader.SkillMiddleware import SkillMiddleware
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_mcp_adapters.sessions import StdioConnection, StreamableHttpConnection
from langchain_mcp_adapters.client import MultiServerMCPClient
from helpers.pdf_injection_middleware import pdf_injection_middleware
from pathlib import Path
from urllib.parse import urlparse

class AgentBuilder:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def skilled(self):
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            caller_file = Path(caller_frame.f_code.co_filename).resolve()
            skills_folder = caller_file.parent / "skills"
        finally:
            del frame
        self.__add_middleware(SkillMiddleware(skills_folder if skills_folder.exists() else None))
        return self
    
    def limited(self, run_limit: int = 12):
        self.__add_middleware(ToolCallLimitMiddleware(run_limit=run_limit, exit_behavior="end"))
        return self
    
    def mcp(self, mcp_servers_locations : dict[str, str | dict]):
        connections = self.__create_connections(mcp_servers_locations)
        client = MultiServerMCPClient(connections)
        tools = asyncio.run(client.get_tools())
        user_tools = self.kwargs.pop("tools", [])
        self.kwargs["tools"] = [*user_tools, *tools]
        return self

    def ask_user_when_needed(self):
        from langchain_core.tools import tool

        @tool
        def ask_user(question: str) -> str:
            """Ask the user for missing information before proceeding."""
            return ""
        
        prompt = self.kwargs.pop("system_prompt", "")
        prompt += "\n\nIf you want to call a tool but you don't have the information necessary for the tool to work (aka arguments for the tool), use ask_user first."
        self.kwargs["system_prompt"] = prompt

        user_tools = self.kwargs.pop("tools", [])
        self.kwargs["tools"] = [*user_tools, ask_user]

        return self.human_in_the_loop(interrupt_on={"ask_user": {"allowed_decisions": ["respond"]}})

    def human_in_the_loop(self, interrupt_on: dict[str, bool | dict] | None = None, description_prefix: str | None = None):
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        hitl_kwargs = {}
        if interrupt_on is not None:
            hitl_kwargs["interrupt_on"] = interrupt_on
        if description_prefix is not None:
            hitl_kwargs["description_prefix"] = description_prefix

        self.__add_middleware(HumanInTheLoopMiddleware(**hitl_kwargs))

        if "checkpointer" not in self.kwargs:
            self.kwargs["checkpointer"] = self.__create_default_checkpointer()

        return self

    def with_memory(self, checkpointer=None):
        if checkpointer is None:
            checkpointer = self.__create_default_checkpointer()
        self.kwargs["checkpointer"] = checkpointer
        return self

    def pdf_reader(self):
        self.__add_middleware(pdf_injection_middleware)
        return self

    def tool_images(self):
        from helpers.tool_image_middleware import relocate_tool_images
        self.__add_middleware(relocate_tool_images)
        return self

    def with_todos(self):
        from langchain.agents.middleware import TodoListMiddleware
        self.__add_middleware(TodoListMiddleware())
        return self

    def build(self):
        return Agent(*self.args, **self.kwargs)

    def __create_connections(self, mcp_servers_locations: dict[str, str | dict]):
        connections = {}
        for name, server_info in mcp_servers_locations.items():
            if isinstance(server_info, str):
                if self._is_file_path(server_info):
                    connection = StdioConnection(transport="stdio", command="python", args=[server_info])
                elif self._is_http_url(server_info):
                    connection = StreamableHttpConnection(transport="http", url=server_info)
                else:
                    raise ValueError(f"Invalid server info: {server_info}")
            else:
                connection = self.__create_connection_from_dict(server_info)
            connections[name] = connection
        return connections

    def __create_connection_from_dict(self, server_info: dict):
        if "args" in server_info:
            if "transport" not in server_info:
                server_info["transport"] = "stdio"
            if "command" not in server_info:
                server_info["command"] = "python"
            if "args" in server_info and not isinstance(server_info["args"], list):
                server_info["args"] = [server_info["args"]]
            return StdioConnection(**server_info)
        elif "url" in server_info:
            if "transport" not in server_info:
                server_info["transport"] = "http"
            return StreamableHttpConnection(**server_info)
        else:
            raise ValueError(f"Invalid server info: {server_info}")
        

    def _is_file_path(self, path: str) -> bool:
        return Path(path).exists()
    
    def _is_http_url(self, url: str) -> bool:
        result = urlparse(url)
        return all([result.scheme, result.netloc])

    def __create_default_checkpointer(self):
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    def __add_middleware(self, *middleware):
        user_middleware = self.kwargs.pop("middleware", None)

        if user_middleware is None:
            user_middleware = tuple()
        elif not isinstance(user_middleware, (list, tuple)):
            user_middleware = [user_middleware]
            
        combined_middleware = [*middleware, *user_middleware]
        self.kwargs["middleware"] = combined_middleware
    