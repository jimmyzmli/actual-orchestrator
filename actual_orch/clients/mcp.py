import os
import asyncio
from contextlib import AsyncExitStack
from mcp.client.sse import sse_client
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
        mcp_endpoint = os.getenv("ACTUAL_MCP_ENDPOINT", "http://localhost:9991")
        if not mcp_endpoint.endswith("/sse"):
            mcp_endpoint = f"{mcp_endpoint.rstrip('/')}/sse"

        print(f"🚀 Connecting to shared Actual Budget MCP server via SSE at {mcp_endpoint}...")
        ctx = sse_client(mcp_endpoint)
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
