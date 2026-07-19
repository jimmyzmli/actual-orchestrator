import asyncio
import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime, timedelta

MCP_SERVER_PATH = "index.js"

async def _find_or_create_transaction_async(date: str, amount: float, payee: str, notes: str = ""):
    server_params = StdioServerParameters(
        command="node",
        args=[MCP_SERVER_PATH],
        env=os.environ.copy()
    )
    
    final_id = None
    target_amount = -int(round(amount * 100)) # Expense is negative in Actual
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 1. Search for existing transaction (+/- 1 day)
                dt = datetime.strptime(date, "%Y-%m-%d")
                start_date = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
                end_date = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
                
                filter_json = json.dumps({
                    "date": {"$gte": start_date, "$lte": end_date}
                })
                query_args = ["run", "--table", "transactions", "--filter", filter_json, "--format", "json"]
                
                res = await session.call_tool("actual_query", arguments={"args": query_args})
                
                if res.content and len(res.content) > 0 and hasattr(res.content[0], 'text'):
                    data = json.loads(res.content[0].text)
                    if isinstance(data, list):
                        for tx in data:
                            tx_amount = tx.get("amount", 0)
                            # Use absolute value for matching to avoid sign issues
                            if abs(tx_amount) == abs(target_amount):
                                print(f"Found matching transaction: {tx.get('id')}")
                                final_id = tx.get("id")
                                break
                                
                if not final_id:
                    # 2. Create if not found
                    accounts_res = await session.call_tool("actual_accounts", arguments={"args": ["list", "--format", "json"]})
                    accounts_data = json.loads(accounts_res.content[0].text)
                    if not accounts_data:
                        raise ValueError("No accounts found in Actual Budget")
                    
                    account_id = accounts_data[0]["id"]
                    
                    tx_data = json.dumps([{
                        "account": account_id,
                        "date": date,
                        "amount": target_amount,
                        "payee_name": payee,
                        "notes": notes
                    }])
                    
                    print(f"Creating new transaction for {date} - ${amount}...")
                    create_args = ["add", "--account", account_id, "--data", tx_data, "--format", "json"]
                    create_res = await session.call_tool("actual_transactions", arguments={"args": create_args})
                    
                    # 'add' command might just return "ok". Re-query to get the new ID.
                    await asyncio.sleep(0.1)
                    res = await session.call_tool("actual_query", arguments={"args": query_args})
                    if res.content and len(res.content) > 0 and hasattr(res.content[0], 'text'):
                        data = json.loads(res.content[0].text)
                        if isinstance(data, list):
                            for tx in data:
                                tx_amount = tx.get("amount", 0)
                                if abs(tx_amount) == abs(target_amount):
                                    final_id = tx.get("id")
                                    break
                    
                    if not final_id:
                        final_id = "unknown_id"
                        
    except Exception as e:
        # The actual budget MCP server often spits out background sync messages to stdout which breaks the SDK's JSONRPC parser 
        # when shutting down, resulting in TaskGroup/BrokenResourceError exceptions. 
        # If we already have the final_id, we can safely ignore it.
        if final_id and ("TaskGroup" in str(e) or "JSONRPCMessage" in str(e)):
            print("Ignored MCP shutdown error (transaction was successfully processed).")
        else:
            print(f"Error communicating with Actual Budget: {e}")
            raise e
            
    return final_id or "unknown_id"

def find_or_create_transaction(date: str, amount: float, payee: str, notes: str = ""):
    return asyncio.run(_find_or_create_transaction_async(date, amount, payee, notes))
