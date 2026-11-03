import os
import json
import threading
import asyncio
from contextlib import AsyncExitStack
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession


def _get_actual_mcp_endpoint():
    mcp_endpoint = os.getenv("ACTUAL_MCP_ENDPOINT") or "http://localhost:9991"
    for suffix in ["/api/mcp/execute", "/messages", "/message", "/sse", "/mcp"]:
        if mcp_endpoint.endswith(suffix):
            mcp_endpoint = mcp_endpoint[:-len(suffix)]
            break
    return f"{mcp_endpoint.rstrip('/')}/mcp"


class _SyncMCPWorker:
    """Dedicated background loop worker for reliable synchronous MCP tool execution."""
    def __init__(self, endpoint_getter):
        self._get_endpoint = endpoint_getter
        self._loop = None
        self._thread = None
        self._started = threading.Event()
        self._session = None
        self._stack = None
        self._lock = None

    def start(self):
        if not self._thread or not self._thread.is_alive():
            self._started.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self._started.wait()

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._lock = asyncio.Lock()
        self._started.set()
        self._loop.run_forever()

    async def _get_session(self):
        async with self._lock:
            if self._session is not None:
                try:
                    await self._session.list_tools()
                    return self._session
                except Exception:
                    if self._stack:
                        try:
                            await self._stack.aclose()
                        except Exception:
                            pass
                    self._session = None
                    self._stack = None

            endpoint = self._get_endpoint()
            self._stack = AsyncExitStack()
            ctx = streamablehttp_client(endpoint, timeout=60.0)
            read, write, _ = await self._stack.enter_async_context(ctx)
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._session = session
            return self._session

    async def _call(self, name: str, args: dict):
        session = await self._get_session()
        result = await session.call_tool(name, args or {})
        if hasattr(result, "content") and result.content:
            for item in result.content:
                if hasattr(item, "text"):
                    try:
                        return json.loads(item.text)
                    except Exception:
                        return item.text
        return result

    async def _close(self):
        if self._lock:
            async with self._lock:
                if self._stack:
                    try:
                        await self._stack.aclose()
                    except Exception:
                        pass
                self._session = None
                self._stack = None

    def call_tool(self, name: str, args: dict = None, timeout: float = 60.0):
        self.start()
        fut = asyncio.run_coroutine_threadsafe(self._call(name, args), self._loop)
        return fut.result(timeout=timeout)

    def close(self):
        if self._loop and self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._close(), self._loop)
            try:
                fut.result(timeout=5.0)
            except Exception:
                pass


class ActualMCPClient:
    _instance = None
    _lock = asyncio.Lock()
    _sync_worker = None

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0  # seconds

    def __init__(self):
        self.stack = AsyncExitStack()
        self.session = None

    @classmethod
    def _get_sync_worker(cls):
        if cls._sync_worker is None:
            cls._sync_worker = _SyncMCPWorker(_get_actual_mcp_endpoint)
        return cls._sync_worker

    @classmethod
    async def get_session(cls):
        loop = asyncio.get_running_loop()
        async with cls._lock:
            if cls._instance is not None:
                if getattr(cls._instance, "_loop", None) != loop:
                    cls._instance = None
                else:
                    # Health-check: verify the cached session is still alive
                    try:
                        await cls._instance.session.list_tools()
                    except Exception:
                        # Session is stale — tear down and reconnect
                        print("⚠️ MCP session health-check failed, reconnecting...")
                        await cls._force_close()

            if cls._instance is None:
                cls._instance = ActualMCPClient()
                cls._instance._loop = loop
                await cls._instance._connect()
            return cls._instance.session

    async def _connect(self):
        mcp_endpoint = _get_actual_mcp_endpoint()

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
        if cls._sync_worker:
            cls._sync_worker.close()

    @classmethod
    async def call_tool(cls, name: str, arguments: dict = None):
        session = await cls.get_session()
        result = await session.call_tool(name, arguments or {})
        if hasattr(result, "content") and result.content:
            for item in result.content:
                if hasattr(item, "text"):
                    try:
                        return json.loads(item.text)
                    except Exception:
                        return item.text
        return result

    @classmethod
    def call_tool_sync(cls, name: str, arguments: dict = None, timeout: float = 60.0):
        return cls._get_sync_worker().call_tool(name, arguments, timeout=timeout)

    # ─── Receipt Convenience Methods (Async) ───
    @classmethod
    async def list_receipts(cls, **kwargs):
        return await cls.call_tool("actual_list_receipts", kwargs)

    @classmethod
    async def get_receipt(cls, **kwargs):
        return await cls.call_tool("actual_get_receipt", kwargs)

    @classmethod
    async def check_receipt_exists(cls, **kwargs):
        res = await cls.call_tool("actual_check_receipt_exists", kwargs)
        return res.get("exists", False) if isinstance(res, dict) else bool(res)

    @classmethod
    async def save_receipt(cls, data: dict):
        return await cls.call_tool("actual_save_receipt", data)

    @classmethod
    async def delete_receipt(cls, **kwargs):
        return await cls.call_tool("actual_delete_receipt", kwargs)

    @classmethod
    async def query_receipts(cls, **kwargs):
        args = ["run", "--table", "receipts"]
        if kwargs.get("select"):
            args.extend(["--select", str(kwargs["select"])])
        if kwargs.get("filter") is not None and str(kwargs["filter"]).strip():
            f = kwargs["filter"]
            args.extend(["--filter", json.dumps(f) if isinstance(f, (dict, list)) else str(f)])
        if kwargs.get("limit") is not None:
            args.extend(["--limit", str(kwargs["limit"])])
        if kwargs.get("offset") is not None:
            args.extend(["--offset", str(kwargs["offset"])])
        if kwargs.get("order_by"):
            args.extend(["--order-by", str(kwargs["order_by"])])
        return await cls.call_tool("actual_query", {"args": args})

    # ─── Receipt Convenience Methods (Sync) ───
    @classmethod
    def list_receipts_sync(cls, **kwargs):
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return cls.call_tool_sync("actual_list_receipts", clean)

    @classmethod
    def get_receipt_sync(cls, **kwargs):
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return cls.call_tool_sync("actual_get_receipt", clean)

    @classmethod
    def check_receipt_exists_sync(cls, **kwargs):
        clean = {k: v for k, v in kwargs.items() if v is not None}
        res = cls.call_tool_sync("actual_check_receipt_exists", clean)
        return res.get("exists", False) if isinstance(res, dict) else bool(res)

    @classmethod
    def save_receipt_sync(cls, data: dict):
        return cls.call_tool_sync("actual_save_receipt", data)

    @classmethod
    def delete_receipt_sync(cls, **kwargs):
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return cls.call_tool_sync("actual_delete_receipt", clean)

    @classmethod
    def query_receipts_sync(cls, **kwargs):
        args = ["run", "--table", "receipts"]
        if kwargs.get("select"):
            args.extend(["--select", str(kwargs["select"])])
        if kwargs.get("filter") is not None and str(kwargs["filter"]).strip():
            f = kwargs["filter"]
            args.extend(["--filter", json.dumps(f) if isinstance(f, (dict, list)) else str(f)])
        if kwargs.get("limit") is not None:
            args.extend(["--limit", str(kwargs["limit"])])
        if kwargs.get("offset") is not None:
            args.extend(["--offset", str(kwargs["offset"])])
        if kwargs.get("order_by"):
            args.extend(["--order-by", str(kwargs["order_by"])])
        return cls.call_tool_sync("actual_query", {"args": args})


class EmailMCPClient:
    """Client for the Email MCP Server (Apple Mail fast querying & rendering)."""
    _instance = None
    _lock = asyncio.Lock()

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0

    def __init__(self):
        self.stack = AsyncExitStack()
        self.session = None

    @classmethod
    async def get_session(cls):
        loop = asyncio.get_running_loop()
        async with cls._lock:
            if cls._instance is not None:
                if getattr(cls._instance, "_loop", None) != loop:
                    cls._instance = None
                else:
                    try:
                        await cls._instance.session.list_tools()
                    except Exception:
                        print("⚠️ Email MCP session health-check failed, reconnecting...")
                        await cls._force_close()

            if cls._instance is None:
                cls._instance = EmailMCPClient()
                cls._instance._loop = loop
                await cls._instance._connect()
            return cls._instance.session

    async def _connect(self):
        mcp_endpoint = os.getenv("EMAIL_MCP_ENDPOINT", "http://localhost:9993/mcp")
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
                try:
                    await self.stack.aclose()
                except Exception:
                    pass
                self.stack = AsyncExitStack()
                self.session = None

                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    print(f"⚠️ Email MCP connection attempt {attempt} failed: {e}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ All {self.MAX_RETRIES} Email MCP connection attempts failed.")

        raise ConnectionError(f"Failed to connect to Email MCP server at {mcp_endpoint} after {self.MAX_RETRIES} attempts: {last_error}")

    @classmethod
    async def _force_close(cls):
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

    @classmethod
    async def call_tool(cls, name: str, arguments: dict = None):
        import json
        session = await cls.get_session()
        result = await session.call_tool(name, arguments or {})
        if hasattr(result, "content") and result.content:
            for item in result.content:
                if hasattr(item, "text"):
                    try:
                        return json.loads(item.text)
                    except Exception:
                        return item.text
        return result

    @classmethod
    async def list_recent_emails(cls, limit: int = 25, since: str = None, mailbox: str = None, account: str = "Outlook"):
        args = {"limit": limit, "account": account}
        if since: args["since"] = since
        if mailbox: args["mailbox"] = mailbox
        return await cls.call_tool("list_recent_emails", args)

    @classmethod
    async def get_email(cls, msg_id: str, format: str = "clean_text", include_pdf_text: bool = True, save_attachments_dir: str = None):
        args = {"msg_id": msg_id, "format": format, "include_pdf_text": include_pdf_text}
        if save_attachments_dir: args["save_attachments_dir"] = save_attachments_dir
        return await cls.call_tool("get_email", args)

    @classmethod
    async def batch_get_emails(cls, msg_ids: list, format: str = "clean_text", include_pdf_text: bool = True):
        return await cls.call_tool("batch_get_emails", {"msg_ids": msg_ids, "format": format, "include_pdf_text": include_pdf_text})

    @classmethod
    async def render_email_screenshot(cls, msg_id: str = None, html_content: str = None, output_path: str = None):
        args = {}
        if msg_id: args["msg_id"] = msg_id
        if html_content: args["html_content"] = html_content
        if output_path: args["output_path"] = output_path
        return await cls.call_tool("render_email_screenshot", args)

