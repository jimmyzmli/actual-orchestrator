import os
from contextlib import asynccontextmanager
from typing import Optional, List
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

@asynccontextmanager
async def compile_workflow(workflow, db_path: str, interrupt_before: Optional[List[str]] = None):
    """
    Compiles a LangGraph workflow with an AsyncSqliteSaver checkpointer.
    Yields both the compiled processor and the raw memory instance.
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(db_path) as memory:
        processor = workflow.compile(
            checkpointer=memory,
            interrupt_before=interrupt_before or []
        )
        yield processor, memory
