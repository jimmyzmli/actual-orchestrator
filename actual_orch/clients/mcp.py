import os
import asyncio
from contextlib import AsyncExitStack
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession


class ActualMCPClient:
    _instance = None
    _lock = asyncio.Lock()

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0  # seconds

    def __init__(self):
        self.stack = AsyncExitStack()
        self.session = None

    @classmethod
    async def get_session(cls):
        async with cls._lock:
            if cls._instance is not None:
                # Health-check: verify the cached session is still alive
                try:
                    await cls._instance.session.list_tools()
                except Exception:
                    # Session is stale — tear down and reconnect
                    print("⚠️ MCP session health-check failed, reconnecting...")
                    await cls._force_close()

            if cls._instance is None:
                cls._instance = ActualMCPClient()
                await cls._instance._connect()
            return cls._instance.session

    async def _connect(self):
        mcp_endpoint = os.getenv("ACTUAL_MCP_ENDPOINT")
        # Strip any known path suffixes to get the base URL
        for suffix in ["/api/mcp/execute", "/messages", "/message", "/sse", "/mcp"]:
            if mcp_endpoint.endswith(suffix):
                mcp_endpoint = mcp_endpoint[:-len(suffix)]
                break
        mcp_endpoint = f"{mcp_endpoint.rstrip('/')}/mcp"

        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                ctx = streamablehttp_client(mcp_endpoint, timeout=60.0)
                read, write, _ = await self.stack.enter_async_context(ctx)

                session_ctx = ClientSession(read, write)
                self.session = await self.stack.enter_async_context(session_ctx)
                await self.session.initialize()
                return
            except Exception as e:
                last_error = e
                # Tear down the stack to clean up any partially-entered contexts
                # from this failed attempt, then create a fresh one for the next try
                try:
                    await self.stack.aclose()
                except Exception:
                    pass
                self.stack = AsyncExitStack()
                self.session = None

                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    print(f"⚠️ Connection attempt {attempt} failed: {e}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ All {self.MAX_RETRIES} connection attempts failed.")

        raise ConnectionError(f"Failed to connect to MCP server at {mcp_endpoint} after {self.MAX_RETRIES} attempts: {last_error}")

    @classmethod
    async def _force_close(cls):
        """Tear down a stale session without acquiring the lock (caller must hold it)."""
        if cls._instance:
            try:
                await cls._instance.stack.aclose()
            except Exception:
                pass
            cls._instance = None

    @classmethod
    async def close_all(cls):
        async with cls._lock:
            await cls._force_close()
