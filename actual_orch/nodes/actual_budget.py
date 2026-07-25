import os
import json
from datetime import datetime, timedelta
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_mcp_adapters.tools import load_mcp_tools

from actual_orch.shared.scoring import score_candidate
from actual_orch.clients.mcp import ActualMCPClient

# For estimate_tokens, just use a simple approximation if tiktoken is not present
def estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4

class TransactionMatchResult(BaseModel):
    matched_id: Optional[str] = Field(description="The UUID of the best matching candidate transaction, if any. Leave null if no good match.")

# ---------------------------------------------------------------------------
# Shared query helper
# ---------------------------------------------------------------------------

async def get_actual_candidates(session, receipt_date: str) -> list[dict]:
    """
    Query Actual Budget for candidate transactions within ±1 days of receipt_date.
    Returns the list of raw/reconstructed candidates, without filtering for #receipt.
    """
    try:
        rd = datetime.strptime(receipt_date, "%Y-%m-%d")
    except ValueError:
        return []

    start_date = (rd - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date   = (rd + timedelta(days=1)).strftime("%Y-%m-%d")

    filter_obj = {
        "date": {"$gte": start_date, "$lte": end_date},
        "account.offbudget": False,
    }

    try:
        res = await session.call_tool("actual_query", {"args": [
            "run", "--table", "transactions",
            "--select", "id,date,amount,payee.name,imported_payee,notes,account.name,is_child,parent_id,transfer_id",
            "--filter", json.dumps(filter_obj),
            "--format", "json",
        ]})
        raw = res.content[0].text if res.content else "[]"
        raw_candidates = json.loads(raw) if isinstance(raw, str) else []
        
        # Reconstruct parents from children since actual_query filters parents out
        parent_map = {}
        candidates = []
        for c in raw_candidates:
            if c.get("is_child") and c.get("parent_id"):
                pid = c["parent_id"]
                if pid not in parent_map:
                    parent_map[pid] = {
                        "id": pid,
                        "date": c.get("date"),
                        "amount": 0,
                        "payee.name": c.get("payee.name"),
                        "imported_payee": c.get("imported_payee"),
                        "notes": c.get("notes") or "",
                        "account.name": c.get("account.name"),
                        "children": []
                    }
                parent_map[pid]["amount"] += c.get("amount", 0)
                parent_map[pid]["children"].append(c)
                # Ensure we catch #receipt if it's on any child
                if "#receipt" in (c.get("notes") or "") and "#receipt" not in parent_map[pid]["notes"]:
                    parent_map[pid]["notes"] += " #receipt"
            else:
                candidates.append(c)
                
        for parent_tx in parent_map.values():
            children = parent_tx.pop("children")
            if parent_tx["amount"] == 0:
                # Do not reconstruct; keep children as separate candidates
                candidates.extend(children)
            else:
                candidates.append(parent_tx)
        
        return candidates
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️ Query for candidates failed: {repr(e)}")
        return []

# ---------------------------------------------------------------------------
# Search Nodes (Deterministic and Agent)
# ---------------------------------------------------------------------------

SCORE_AUTO_ACCEPT = 0.90

async def search_deterministic_node(state: dict):
    """
    Node 1: Deterministic matching and tiebreaker.
    """
    try:
        print("Matching transaction in Actual Budget (Deterministic)...")
        parsed = state.get("parsed_data")
        if not parsed:
            return {"status": "error", "error": "No parsed data found"}
            
        date = parsed["date"]
        amount = parsed["total"]
        payee = parsed["payee"]
        receipt_cents = abs(int(amount * 100))
        
        session = await ActualMCPClient.get_session()
        
        candidates = await get_actual_candidates(session, date)
        if not candidates:
            print("  No candidates found in date range, falling through to agent...")
            return {"actual_tx_id": "unknown_id", "status": "fallback_to_agent"}
            
        # Score and filter
        scored = []
        for c in candidates:
            c["match_score"] = score_candidate(payee, receipt_cents, date, c)
            scored.append(c)
            
        scored.sort(key=lambda x: x["match_score"], reverse=True)
        
        if not scored:
            print("  All candidates already tagged or none valid, falling through to agent...")
            return {"actual_tx_id": "unknown_id", "status": "fallback_to_agent"}
            
        top = scored[0]
        top_score = top["match_score"]
        print(f"  Top candidate: {top['id']} "
              f"(score={top_score:.3f}, payee={top.get('payee.name', '?')}, "
              f"amount=${abs(top.get('amount', 0))/100:.2f})")
              
        final_id = "unknown_id"
        
        if top_score >= SCORE_AUTO_ACCEPT:
            final_id = top["id"]
            print(f"✅ Auto-matched (score {top_score:.3f}): {final_id}")
        else:
            print(f"  No candidate scored ≥ {SCORE_AUTO_ACCEPT}, falling through to agent...")
            
        # Validate deterministic match
        if final_id != "unknown_id":
            matched_candidate = next((c for c in scored if c["id"] == final_id), None)
            if matched_candidate:
                actual_amt = abs(matched_candidate.get("amount", 0))
                min_allowed = receipt_cents * 0.90
                max_allowed = receipt_cents * 1.30
                if not (min_allowed <= actual_amt <= max_allowed):
                    print(f"❌ VALIDATION FAILED: Bank amount ${actual_amt/100:.2f} "
                          f"outside 0.90x–1.30x of receipt ${amount:.2f}. Rejecting.")
                    final_id = "unknown_id"
                    
        if final_id == "unknown_id":
            return {"actual_tx_id": "unknown_id", "status": "fallback_to_agent"}
            
        transfer_id = None
        matched = next((c for c in scored if c["id"] == final_id), None)
        if matched and matched.get("transfer_id"):
            transfer_id = matched.get("transfer_id")
            
        return {"actual_tx_id": final_id, "actual_tx_id_transfer": transfer_id}
        
    except Exception as e:
        if "TaskGroup" in str(e) or "JSONRPCMessage" in str(e) or "BrokenResourceError" in str(e):
            print("Ignored MCP shutdown error, agent succeeded.")
            return {"actual_tx_id": "unknown_id", "status": "fallback_to_agent"}
        else:
            return {"status": "error", "error": f"Deterministic search failed: {str(e)}"}


async def search_agent_node(state: dict):
    """
    Node 2: Agent fallback. Runs when deterministic matching fails.
    """
    try:
        print("  🔄 Falling back to LLM structured selection (Agent Node)...")
        parsed = state.get("parsed_data")
        if not parsed:
            return {"status": "error", "error": "No parsed data found"}
            
        date = parsed["date"]
        amount = parsed["total"]
        payee = parsed["payee"]
        receipt_cents = abs(int(amount * 100))
        
        session = await ActualMCPClient.get_session()
        
        candidates = await get_actual_candidates(session, date)
        
        clean_candidates = candidates
        
        narrowed_candidates = []
        for c in clean_candidates:
            narrowed_candidates.append({
                "id": c.get("id"),
                "date": c.get("date"),
                "amount": c.get("amount"),
                "payee": c.get("payee.name") or c.get("imported_payee"),
                "account": c.get("account.name"),
                "notes": (c.get("notes") or "")[:200]
            })
            
        def get_sort_key(c):
            try:
                cd = datetime.strptime(c.get("date"), "%Y-%m-%d")
                rd = datetime.strptime(date, "%Y-%m-%d")
                dd = abs((cd - rd).days)
            except Exception:
                dd = 999
            
            c_amt = abs(c.get("amount", 0))
            a_diff = abs(c_amt - receipt_cents)
            return (dd, a_diff)
                
        narrowed_candidates.sort(key=get_sort_key)
        narrowed_candidates = narrowed_candidates[:50]
        
        parse_retries = state.get("parse_retries", 0)
        model_name = "deepseek-v4-pro" if parse_retries > 0 else "deepseek-v4-flash"
        
        print(f"  🔄 Retrying match with model ({model_name})...")
        
        prompt_text = f"""You are a financial matching assistant.

Receipt details:
{json.dumps(parsed, indent=2)}

Candidate transactions from the bank:
{json.dumps(narrowed_candidates, indent=2)}

Task: Identify the transaction that matches this receipt. You must respond in JSON.
Rules:
- Amounts must be exactly equal OR very close (e.g., tip included, currency difference).
- Payee names might differ but should be conceptually similar.
- Date must be within 1 day.

Respond in JSON format with the 'matched_id' of the matching transaction, or null if no match exists."""
            
        llm = ChatOpenAI(
            model=model_name, 
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""), 
            base_url="https://api.deepseek.com/v1"
        )
        
        structured_llm = llm.with_structured_output(TransactionMatchResult, method="json_mode")
        
        if state.get("log_agent"):
            print(f"\n--- AGENT PROMPT ({model_name}) ---")
            print(prompt_text)
            print("---------------------------------------\n")
            
        final_id = "unknown_id"
        try:
            result = await structured_llm.ainvoke([HumanMessage(content=prompt_text)])
            
            if state.get("log_agent"):
                print(f"\n--- AGENT RESPONSE ({model_name}) ---")
                print(result.dict())
                print("---------------------------------------\n")
                
            if result.matched_id:
                final_id = result.matched_id
                print(f"Agent matched tx: {final_id} with {model_name}")
                # Validate
                matched = next((c for c in narrowed_candidates if c["id"] == final_id), None)
                if matched:
                    print(f"✅ Agent selected valid candidate: {final_id}")
                    
                    # Validate amount
                    actual_amt = abs(matched.get("amount", 0))
                    min_allowed = receipt_cents * 0.90
                    max_allowed = receipt_cents * 1.30
                    if not (min_allowed <= actual_amt <= max_allowed):
                        print(f"❌ VALIDATION FAILED: Agent amount ${actual_amt/100:.2f} "
                              f"outside 0.90x–1.30x of receipt ${amount:.2f}. Rejecting.")
                        final_id = "unknown_id"
                        
                    # Validate date
                    if final_id != "unknown_id" and matched.get("date"):
                        try:
                            rd = datetime.strptime(date, "%Y-%m-%d")
                            cd = datetime.strptime(matched.get("date"), "%Y-%m-%d")
                            if abs((rd - cd).days) > 1:
                                print(f"❌ VALIDATION FAILED: Agent date {matched.get('date')} "
                                      f"is >1 days from receipt {date}. Rejecting.")
                                final_id = "unknown_id"
                        except Exception:
                            pass
                else:
                    print(f"❌ VALIDATION FAILED: Agent claimed to find tx {final_id}, but it was not in the narrowed candidates.")
                    final_id = "unknown_id"
            else:
                print(f"Agent ({model_name}) output didn't contain matched_id")
                final_id = "unknown_id"
        except Exception as e:
            print(f"Warning: Failed to parse structured output from {model_name}: {e}")
            final_id = "unknown_id"

        if final_id == "unknown_id":
            if state.get("parse_retries", 0) == 0:
                print(f"  ❌ Matching failed completely. Triggering parse_receipt retry...")
                return {"parse_retries": 1, "status": "retry_parse", "actual_tx_id": "unknown_id"}
            else:
                print(f"  ❌ Matching failed completely after retry. Routing to human approval.")
                return {"status": "human_approval", "actual_tx_id": "unknown_id"}
                
        transfer_id = None
        matched_full = next((c for c in clean_candidates if c["id"] == final_id), None)
        if matched_full and matched_full.get("transfer_id"):
            transfer_id = matched_full.get("transfer_id")
            
        return {"actual_tx_id": final_id, "actual_tx_id_transfer": transfer_id}
        
    except Exception as e:
        if "TaskGroup" in str(e) or "JSONRPCMessage" in str(e) or "BrokenResourceError" in str(e):
            print("Ignored MCP shutdown error, agent succeeded.")
            return {"status": "human_approval", "actual_tx_id": "unknown_id"}
        else:
            return {"status": "error", "error": f"Agent search failed: {str(e)}"}

async def tag_actual_transaction_node(state: dict):
    """Append #receipt to the matched/created transaction's notes in Actual Budget."""
    tx_id = state.get("actual_tx_id")
    if not tx_id or tx_id == "unknown_id":
        return {}

    try:
        print(f"Tagging transaction {tx_id} with #receipt...")
        session = await ActualMCPClient.get_session()
        
        # Fetch initial transaction
        target_tx_id = tx_id
        existing_notes = ""
        is_child = False
        parent_id = None
        
        try:
            res = await session.call_tool("actual_query", {"args": [
                "run", "--table", "transactions", "--filter", json.dumps({"id": tx_id}), 
                "--select", "id,notes,is_parent,is_child,parent_id,account,date", "--format", "json"
            ]})
            if res.content and len(res.content) > 0:
                query_data = json.loads(res.content[0].text)
                if query_data and len(query_data) > 0:
                    tx_data = query_data[0]
                    existing_notes = tx_data.get("notes") or ""
                    is_child = tx_data.get("is_child")
                    parent_id = tx_data.get("parent_id")
                else:
                    # It might be a parent transaction (actual_query filters them out)
                    # Try to find its children to get account and date
                    c_res = await session.call_tool("actual_query", {"args": [
                        "run", "--table", "transactions", "--filter", json.dumps({"parent_id": tx_id}), 
                        "--select", "account,date", "--format", "json"
                    ]})
                    if c_res.content and len(c_res.content) > 0:
                        c_data = json.loads(c_res.content[0].text)
                        if c_data and len(c_data) > 0:
                            tx_data = c_data[0]
                            is_child = True # Force resolution block
                            parent_id = tx_id
        except Exception as e:
            print(f"Failed to fetch transaction details: {e}")
            
        # If it's a child, resolve to parent
        if is_child and parent_id:
            print(f"Matched child transaction. Resolving to parent {parent_id}")
            target_tx_id = parent_id
            # Fetch parent notes and split breakdown using list API
            try:
                acc = tx_data.get("account")
                dt = tx_data.get("date")
                if acc and dt:
                    list_res = await session.call_tool("actual_transactions", {"args": [
                        "list", "--account", acc,
                        "--start", dt, "--end", dt,
                        "--format", "json"
                    ]})
                    if list_res.content and len(list_res.content) > 0:
                        txs = json.loads(list_res.content[0].text)
                        for t in txs:
                            if t.get("id") == target_tx_id:
                                existing_notes = t.get("notes") or ""
                                break
            except Exception as e:
                print(f"Failed to fetch parent notes: {e}")
        
        # Append #receipt to existing notes
        if "#receipt" not in existing_notes:
            if existing_notes.strip():
                new_notes = f"{existing_notes.strip()} #receipt"
            else:
                new_notes = "#receipt"
        else:
            print(f"Transaction {target_tx_id} already has #receipt tag.")
            # Return potentially updated tx_id if we resolved to parent
            if target_tx_id != tx_id:
                return {"actual_tx_id": target_tx_id}
            return {}

        mcp_tools = await load_mcp_tools(session)
        tx_tool = next((t for t in mcp_tools if t.name == "actual_transactions"), None)
        if tx_tool:
            await tx_tool.ainvoke({"args": [
                "update", target_tx_id,
                "--data", json.dumps({"notes": new_notes})
            ]})
            print(f"✅ Tagged transaction {target_tx_id} with #receipt")
            
        if target_tx_id != tx_id:
            return {"actual_tx_id": target_tx_id}
            
    except Exception as e:
        if "TaskGroup" in str(e) or "JSONRPCMessage" in str(e) or "BrokenResourceError" in str(e):
            print("Ignored MCP shutdown error, tag succeeded.")
        else:
            print(f"⚠️ Failed to tag transaction (non-fatal): {e}")
    return {}
