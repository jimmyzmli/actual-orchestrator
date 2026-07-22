import os
import asyncio
from contextlib import AsyncExitStack
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession


class ActualMCPClient:
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.stack = AsyncExitStack()
        self.session = None

    @classmethod
    async def get_session(cls):
        async with cls._lock:
            if cls._instance is None:
                cls._instance = ActualMCPClient()
                await cls._instance._connect()
            return cls._instance.session

    async def _connect(self):
        mcp_cli = os.getenv("ACTUAL_MCP_CLI", "actual-mcp")

        import shlex
        parts = shlex.split(mcp_cli)
        server_params = StdioServerParameters(
            command=parts[0],
            args=parts[1:],
            env=os.environ.copy(),
        )
        print("🚀 Starting shared Actual Budget MCP server instance via stdio...")
        ctx = stdio_client(server_params)
        read, write = await self.stack.enter_async_context(ctx)

        session_ctx = ClientSession(read, write)
        self.session = await self.stack.enter_async_context(session_ctx)
        await self.session.initialize()

    @classmethod
    async def close_all(cls):
        async with cls._lock:
            if cls._instance:
                try:
                    await cls._instance.stack.aclose()
                except Exception:
                    pass
                cls._instance = None
