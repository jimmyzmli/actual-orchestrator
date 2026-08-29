import json
from actual_orch.clients.mcp import ActualMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

async def remove_transaction_tag(tx_id: str, tag: str = "#receipt") -> bool:
    """
    Removes a specific tag from an Actual Budget transaction's notes.
    Handles child-to-parent transaction resolution automatically.
    """
    try:
        session = await ActualMCPClient.get_session()
        
        res = await session.call_tool("actual_query", {"args": [
            "run", "--table", "transactions", "--filter", json.dumps({"id": tx_id}),
            "--select", "id,notes,is_parent,is_child,parent_id,account,date", "--format", "json"
        ]})
        
        target_tx_id = tx_id
        existing_notes = ""
        is_child = False
        parent_id = None
        
        if res.content and len(res.content) > 0:
            query_data = json.loads(res.content[0].text)
            if query_data and len(query_data) > 0:
                tx_data = query_data[0]
                existing_notes = tx_data.get("notes") or ""
                is_child = tx_data.get("is_child")
                parent_id = tx_data.get("parent_id")
            else:
                c_res = await session.call_tool("actual_query", {"args": [
                    "run", "--table", "transactions", "--filter", json.dumps({"parent_id": tx_id}), 
                    "--select", "account,date", "--format", "json"
                ]})
                if c_res.content and len(c_res.content) > 0:
                    c_data = json.loads(c_res.content[0].text)
                    if c_data and len(c_data) > 0:
                        tx_data = c_data[0]
                        is_child = True
                        parent_id = tx_id
        
        if is_child and parent_id:
            target_tx_id = parent_id
            try:
                acc = tx_data.get("account")
                dt = tx_data.get("date")
                if acc and dt:
                    list_res = await session.call_tool("actual_list_transactions", {"args": [
                        "--account", acc,
                        "--start", dt, "--end", dt,
                        "--format", "json"
                    ]})
                    if list_res.content and len(list_res.content) > 0:
                        txs = json.loads(list_res.content[0].text)
                        for t in txs:
                            if t.get("id") == target_tx_id:
                                existing_notes = t.get("notes") or ""
                                break
            except Exception:
                pass
        
        if tag in existing_notes:
            new_notes = existing_notes.replace(tag, "").replace("  ", " ").strip()
            mcp_tools = await load_mcp_tools(session)
            tx_tool = next((t for t in mcp_tools if t.name in ("actual_update_transaction", "actual_transactions")), None)
            
            if tx_tool:
                await tx_tool.ainvoke({"args": [
                    target_tx_id,
                    "--data", json.dumps({"notes": new_notes})
                ]})
                print(f"Removed {tag} tag from Actual Budget transaction {target_tx_id}.")
                return True
        else:
            print(f"Actual Budget transaction {target_tx_id} does not have {tag} tag.")
            return False
            
    except Exception as e:
        if "TaskGroup" in str(e) or "JSONRPCMessage" in str(e) or "BrokenResourceError" in str(e):
            pass
        else:
            print(f"Failed to remove tag from Actual Budget: {e}")
        return False
    finally:
        await ActualMCPClient.close_all()
