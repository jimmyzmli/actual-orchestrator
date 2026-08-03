import os
import json
import re
from datetime import datetime, timedelta
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_mcp_adapters.tools import load_mcp_tools

from actual_orch.shared.scoring import score_candidate
from actual_orch.clients.mcp import ActualMCPClient

def strip_emojis(s: str) -> str:
    if not s:
        return "" if s is None else str(s)
    pattern = re.compile(r'[\u2600-\u27BF\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2300-\u23FF]+')
    return pattern.sub('', s).strip()

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
                # Include the reconstructed parent AND individual children
                candidates.append(parent_tx)
                candidates.extend(children)
        
        # Try to fetch real parent data to replace reconstructed approximations
        if parent_map:
            try:
                # Use a split child's account to query via list API for real parent records
                sample_parent = next(iter(parent_map.values()))
                acc = sample_parent.get("account.name")
                if acc:
                    # Resolve account name to ID for list API
                    acc_res = await session.call_tool("actual_accounts", {"args": ["list", "--format", "json"]})
                    acc_id = None
                    if acc_res.content:
                        accts = json.loads(acc_res.content[0].text)
                        acc_match = next((a for a in accts if a.get("name") == acc), None)
                        if acc_match:
                            acc_id = acc_match.get("id")
                    if acc_id:
                        list_res = await session.call_tool("actual_transactions", {"args": [
                            "list", "--account", acc_id,
                            "--start", start_date, "--end", end_date,
                            "--format", "json"
                        ]})
                        if list_res.content:
                            all_txs = json.loads(list_res.content[0].text)
                            for tx in all_txs:
                                pid = tx.get("id")
                                if pid in parent_map:
                                    # Replace reconstructed parent with real data
                                    for i, c in enumerate(candidates):
                                        if c.get("id") == pid and not c.get("is_child"):
                                            candidates[i] = {
                                                "id": pid,
                                                "date": tx.get("date", c.get("date")),
                                                "amount": tx.get("amount", c.get("amount", 0)),
                                                "payee.name": tx.get("payee", c.get("payee.name")),
                                                "imported_payee": tx.get("imported_payee", c.get("imported_payee")),
                                                "notes": tx.get("notes") or c.get("notes", ""),
                                                "account.name": acc,
                                            }
                                            break
            except Exception as e:
                pass  # Fall back to reconstructed parents silently
        
        return candidates
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️ Query for candidates failed: {repr(e)}")
        raise e

# ---------------------------------------------------------------------------
# Search Nodes (Deterministic and Agent)
# ---------------------------------------------------------------------------

SCORE_AUTO_ACCEPT = 0.80

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
                
                if state.get("source") == "email" and actual_amt != receipt_cents:
                    print(f"❌ VALIDATION FAILED: Email receipt amount ${amount:.2f} "
                          f"must exactly match bank amount ${actual_amt/100:.2f}. Rejecting.")
                    final_id = "unknown_id"
                else:
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
            amount_cents = c.get("amount", 0)
            amount_dollars = amount_cents / 100.0 if amount_cents else 0
            entry = {
                "id": c.get("id"),
                "date": c.get("date"),
                "amount": amount_dollars,
                "payee": strip_emojis(c.get("payee.name") or c.get("imported_payee")),
                "account": strip_emojis(c.get("account.name")),
                "notes": (c.get("notes") or "")[:200],
            }
            if c.get("is_child"):
                entry["is_child"] = True
                entry["parent_id"] = c.get("parent_id")
            narrowed_candidates.append(entry)
            
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
        
        parse_attempts = state.get("parse_attempts", [])
        already_retried = any(a.get("model") == "search_agent" for a in parse_attempts)
        
        if not narrowed_candidates:
            print("  ❌ No candidates available for agent to search. Failing immediately.")
            if not already_retried:
                print(f"  ❌ Matching failed completely. Triggering parse_receipt retry...")
                return {
                    "parse_attempts": parse_attempts + [{"model": "search_agent", "success": False, "error": "No candidates"}],
                    "status": "retry_parse",
                    "actual_tx_id": "unknown_id"
                }
            else:
                print(f"  ❌ Matching failed completely after retry. Routing to human approval.")
                return {"status": "human_approval", "actual_tx_id": "unknown_id"}
        
        model_name = "deepseek-v4-pro" if already_retried else "deepseek-v4-flash"
        
        print(f"  🔄 Retrying match with model ({model_name})...")
        
        prompt_text = f"""You are a financial matching assistant.

Receipt details:
{json.dumps(parsed, indent=2)}

Candidate transactions from the bank:
{json.dumps(narrowed_candidates, indent=2)}

Task: Identify the transaction that matches this receipt. You must respond in JSON.
Rules:
- Amounts must be exactly equal OR very close (e.g., tip included, currency difference).
- Split transaction children (is_child=true) are valid candidates. Match on the individual child amount, not the parent total.
- Payee names might differ but should be conceptually similar.
- Date must be within 1 day.

Respond in JSON format with the 'matched_id' of the matching transaction, or null if no match exists."""
            
        llm = ChatOpenAI(
            model=model_name, 
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""), 
            base_url="https://api.deepseek.com/v1"
        )
        
        structured_llm = llm.with_structured_output(TransactionMatchResult, method="json_mode")
        
        if state.get("flag_log_agent"):
            print(f"\n--- AGENT PROMPT ({model_name}) ---")
            print(prompt_text)
            print("---------------------------------------\n")
            
        final_id = "unknown_id"
        try:
            result = await structured_llm.ainvoke([HumanMessage(content=prompt_text)])
            
            if state.get("flag_log_agent"):
                print(f"\n--- AGENT RESPONSE ({model_name}) ---")
                print(result.dict())
                print("---------------------------------------\n")
                
            if result.matched_id:
                final_id = result.matched_id
                print(f"Agent matched tx: {final_id} with {model_name}")
                # Validate
                matched = next((c for c in clean_candidates if c["id"] == final_id), None)
                if matched:
                    print(f"✅ Agent selected valid candidate: {final_id}")
                    
                    # Validate amount (clean_candidates has amount in cents)
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
                    print(f"❌ VALIDATION FAILED: Agent claimed to find tx {final_id}, but it was not in the candidates.")
                    final_id = "unknown_id"
            else:
                print(f"Agent ({model_name}) output didn't contain matched_id")
                final_id = "unknown_id"
        except Exception as e:
            print(f"Warning: Failed to parse structured output from {model_name}: {e}")
            final_id = "unknown_id"

        if final_id == "unknown_id":
            if not already_retried:
                print(f"  ❌ Matching failed completely. Triggering parse_receipt retry...")
                return {
                    "parse_attempts": parse_attempts + [{"model": "search_agent", "success": False, "error": "Match failed"}],
                    "status": "retry_parse",
                    "actual_tx_id": "unknown_id"
                }
            else:
                print(f"  ❌ Matching failed completely after retry. Routing to human approval.")
                return {"status": "human_approval", "actual_tx_id": "unknown_id"}
                
        transfer_id = None
        matched_full = next((c for c in clean_candidates if c["id"] == final_id), None)
        if matched_full and matched_full.get("transfer_id"):
            transfer_id = matched_full.get("transfer_id")
            
        return {"actual_tx_id": final_id, "actual_tx_id_transfer": transfer_id}
        
    except Exception as e:
        return {"status": "error", "error": f"Agent search failed: {str(e)}"}

async def save_actual_transaction_node(state: dict):
    """
    Node 3: Auto-create the transaction if deterministic and agent matching fail.
    Only triggers if flag_create_missing is True.
    """
    if state.get("actual_tx_id") and state.get("actual_tx_id") != "unknown_id":
        return {}
        
    try:
        print("  ➕ Auto-creating missing transaction (save_actual_transaction_node)...")
        parsed = state.get("parsed_data")
        if not parsed:
            return {"status": "error", "error": "No parsed data found for auto-creation"}
            
        date = parsed["date"]
        payee = parsed["payee"]
        
        proposed_txs = state.get("proposed_transactions") or []
        
        # Calculate total from proposed transactions (in CAD — FX already applied by parse node)
        amount = parsed.get("total", 0.0)
        if not amount and proposed_txs:
            amount = sum(pt.get("amount", 0.0) for pt in proposed_txs)
        
        # Load account-mapping.json
        import os
        import json
        mapping_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../actual-utils/cache/account-mapping.json"))
        
        mapping = {}
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
                
        if not mapping:
            print("  ⚠️ account-mapping.json not found or empty.")
            return {"status": "human_approval"}
            
        selected_key = parsed.get("account_key")
        if not selected_key or selected_key not in mapping:
            print(f"  ⚠️ Parsed account key '{selected_key}' is invalid or missing. Falling back to human approval.")
            return {"status": "human_approval"}
            
        account_id = mapping[selected_key]
        print(f"  ✅ Extracted account key '{selected_key}' -> {account_id}")
            
        session = await ActualMCPClient.get_session()
        
        # Create transaction — receipts are expenses (positive amounts) so we negate them for the API.
        # Refunds are negative amounts, negating them makes them positive (inflow) for the API.
        expense_cents = int(round(-amount * 100))
        parsed_notes = (parsed.get("notes") or "").strip()
        tx_notes = f"{parsed_notes} #receipt".strip() if parsed_notes else "#receipt"
        
        # Load categories for reverse lookup (name/id -> id)
        cats = []
        try:
            categories_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../actual-utils/cache/categories.json"))
            if os.path.exists(categories_path):
                with open(categories_path, "r") as f:
                    loaded_cats = json.load(f)
                    if isinstance(loaded_cats, list):
                        cats = loaded_cats
        except Exception:
            pass
        
        if len(proposed_txs) > 1:
            print("  ➕ Detected multiple proposed transactions. Creating split transaction...")
            new_tx = {
                "date": date,
                "amount": expense_cents,
                "payee_name": payee,
                "notes": tx_notes,
                "cleared": False,
                "subtransactions": []
            }
            for pt in proposed_txs:
                # Negate: positive pt_amount (expense) -> negative cents for Actual API
                sub_cents = int(round(-pt.get("amount", 0.0) * 100))
                sub = {
                    "amount": sub_cents,
                    "notes": pt.get("notes", "")
                }
                pt_cat = pt.get("category")
                if pt_cat:
                    cat_match = next((c for c in cats if c.get("name") == pt_cat or c.get("id") == pt_cat), None)
                    sub["category"] = cat_match.get("id") if cat_match else pt_cat
                new_tx["subtransactions"].append(sub)
        else:
            new_tx = {
                "date": date,
                "amount": expense_cents,
                "payee_name": payee,
                "notes": tx_notes,
                "cleared": False
            }
            # Single transaction — get category from the first proposed transaction
            if proposed_txs:
                pt_cat = proposed_txs[0].get("category")
                if pt_cat:
                    cat_match = next((c for c in cats if c.get("name") == pt_cat or c.get("id") == pt_cat), None)
                    if cat_match:
                        new_tx["category"] = cat_match.get("id")
        
        # Save using Actual MCP
        mcp_tools = await load_mcp_tools(session)
        tx_tool = next((t for t in mcp_tools if t.name == "actual_transactions"), None)
        if tx_tool:
            await tx_tool.ainvoke({"args": [
                "add", "--account", account_id,
                "--data", json.dumps([new_tx])
            ]})
            print(f"  ✅ Created transaction for {payee} on {date} for {amount}")
            
            # Query back the ID
            filter_obj = {
                "account": account_id,
                "date": date
            }
            q_res = await session.call_tool("actual_query", {"args": [
                "run", "--table", "transactions",
                "--filter", json.dumps(filter_obj),
                "--select", "id,amount,payee.name,notes,is_child,parent_id",
                "--format", "json"
            ]})
            if q_res.content and len(q_res.content) > 0:
                txs = json.loads(q_res.content[0].text)
                if txs and len(txs) > 0:
                    # 1. Try to find the exact parent transaction match (for single/normal tx)
                    for tx in txs:
                        if not tx.get("is_child") and tx.get("amount") == expense_cents and tx.get("notes") and "#receipt" in tx["notes"]:
                            print(f"  ✅ Retrieved new transaction ID: {tx['id']}")
                            return {"actual_tx_id": tx["id"], "status": "continue"}
                            
                    # 2. Try to find a child transaction match (for split tx)
                    for tx in txs:
                        if tx.get("is_child") and tx.get("parent_id"):
                            # Match if payee is the same and amount matches one of the subtransactions
                            sub_amounts = [int(round(-pt.get("amount", 0.0) * 100)) for pt in proposed_txs]
                            if (payee and tx.get("payee.name") == payee) or (tx.get("amount") in sub_amounts):
                                print(f"  ✅ Retrieved new split transaction parent ID: {tx['parent_id']}")
                                return {"actual_tx_id": tx["parent_id"], "status": "continue"}
                                
                    # 3. Fallback to first returned transaction
                    fallback_tx = txs[0]
                    fallback_id = fallback_tx.get("parent_id") if fallback_tx.get("is_child") else fallback_tx["id"]
                    print(f"  ✅ Retrieved new transaction ID (fallback): {fallback_id}")
                    return {"actual_tx_id": fallback_id, "status": "continue"}
            
            print("  ⚠️ Created transaction but couldn't query its ID. PocketBase won't have the ID.")
            return {"actual_tx_id": "unknown_id", "status": "continue"}
            
    except Exception as e:
        print(f"  ❌ Auto-create failed: {e}")
        return {"status": "error", "error": f"Auto-create failed: {str(e)}"}
    return {}

async def tag_actual_transaction_node(state: dict):
    """Append #receipt to the matched/created transaction's notes in Actual Budget."""
    tx_id = state.get("actual_tx_id")
    if not tx_id or tx_id == "unknown_id":
        return {}

    try:
        print(f"Tagging transaction {tx_id} with #receipt...")
        session = await ActualMCPClient.get_session()
        
        # Fetch the transaction to get its current notes and full data
        existing_notes = ""
        tx_data = None
        
        try:
            res = await session.call_tool("actual_query", {"args": [
                "run", "--table", "transactions", "--filter", json.dumps({"id": tx_id}), 
                "--select", "id,date,amount,payee,imported_payee,notes,category,account,cleared,is_child,parent_id,sort_order",
                "--format", "json"
            ]})
            if res.content and len(res.content) > 0:
                query_data = json.loads(res.content[0].text)
                if query_data and len(query_data) > 0:
                    tx_data = query_data[0]
                    existing_notes = tx_data.get("notes") or ""
                    if tx_data.get("is_child"):
                        print(f"Matched child transaction {tx_id} (parent: {tx_data.get('parent_id')}). Tagging child directly.")
                # else: tx_id is a parent row (actual_query filters these out) — tag it directly anyway
        except Exception as e:
            print(f"Failed to fetch transaction details: {e}")
        
        # Append #receipt to existing notes
        if "#receipt" not in existing_notes:
            if existing_notes.strip():
                new_notes = f"{existing_notes.strip()} #receipt"
            else:
                new_notes = "#receipt"
        else:
            print(f"Transaction {tx_id} already has #receipt tag.")
            return {}

        # Build update payload with only writable fields to prevent the update API
        # from choking on relational/read-only fields returned by actual_query
        # (e.g. payee.name, account.name, is_child, parent_id, transfer_id).
        # This also preserves existing child transaction fields that would otherwise
        # be wiped out if we only sent {"notes": ...}.
        WRITABLE_FIELDS = {"date", "amount", "payee", "payee_name", "imported_payee",
                           "notes", "category", "account", "cleared", "imported_id", "sort_order"}
        update_payload = {"notes": new_notes}
        if tx_data:
            update_payload = {k: v for k, v in tx_data.items() if k in WRITABLE_FIELDS}
            update_payload["notes"] = new_notes

        mcp_tools = await load_mcp_tools(session)
        tx_tool = next((t for t in mcp_tools if t.name == "actual_transactions"), None)
        if tx_tool:
            await tx_tool.ainvoke({"args": [
                "update", tx_id,
                "--data", json.dumps(update_payload)
            ]})
            print(f"✅ Tagged transaction {tx_id} with #receipt")
            
    except Exception as e:
        print(f"⚠️ Failed to tag transaction (non-fatal): {e}")
    return {}
