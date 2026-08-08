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
    pattern = re.compile(r'[\u2600-\u27BF\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2300-\u23FF\u2B00-\u2BFF\uFE00-\uFE0F]+')
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
    updated_amount: Optional[float] = Field(default=None, description="If allowed_to_update is true and the receipt has new info (like tip), output the new total amount here. The polarity must match the receipt (positive for expense, negative for deposit). Otherwise null.")
    updated_payee: Optional[str] = Field(default=None, description="If allowed_to_update is true and the receipt has a better payee, output it here. Otherwise null.")
    updated_notes: Optional[str] = Field(default=None, description="If allowed_to_update is true and you want to append/update notes, output them here. Otherwise null.")
    updated_transfer_account: Optional[str] = Field(default=None, description="If allowed_to_update is true and the receipt indicates it is a transfer to another account, output the exact transfer account name here. Otherwise null.")

# ---------------------------------------------------------------------------
# Shared query helper
# ---------------------------------------------------------------------------

def print_matched_transaction(tx_data: dict, existing_notes: str = None):
    if not tx_data:
        return
    payee_name = tx_data.get("payee.name") or tx_data.get("imported_payee") or "?"
    account_name = tx_data.get("account.name") or "?"
    amount_disp = abs(tx_data.get("amount", 0)) / 100.0
    date_disp = tx_data.get("date") or "?"
    if existing_notes is None:
        existing_notes = tx_data.get("notes") or ""
    notes_disp = (existing_notes or "").strip()
    
    print(f"  \n  📋 Matched transaction:")
    print(f"     Date:    {date_disp}")
    print(f"     Payee:   {payee_name}")
    print(f"     Amount:  ${amount_disp:.2f}")
    print(f"     Account: {account_name}")
    if notes_disp:
        print(f"     Notes:   {notes_disp[:100]}")

async def get_actual_candidates(session, receipt_date: str) -> list[dict]:
    """
    Query Actual Budget for candidate transactions within ±1 days of receipt_date.
    Returns the list of raw/reconstructed candidates, without filtering for #receipt.
    """
    try:
        rd = datetime.strptime(receipt_date, "%Y-%m-%d")
    except ValueError:
        return []

    # Default ±1 day range
    start_date = (rd - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date   = (rd + timedelta(days=1)).strftime("%Y-%m-%d")

    # Additional requested dates
    first_day_this_month = rd.replace(day=1)
    last_day_last_month = (first_day_this_month - timedelta(days=1)).strftime("%Y-%m-%d")
    
    if rd.month == 12:
        first_day_next_month = rd.replace(year=rd.year+1, month=1, day=1)
    else:
        first_day_next_month = rd.replace(month=rd.month+1, day=1)
    first_day_next_month_str = first_day_next_month.strftime("%Y-%m-%d")

    target_dates = list(set([
        rd.strftime("%Y-%m-%d"),
        start_date,
        end_date,
        last_day_last_month,
        first_day_next_month_str
    ]))

    raw_candidates = []
    
    filter_obj = {
        "$or": [{"date": d} for d in target_dates],
        "account.offbudget": False,
    }

    try:
        res = await session.call_tool("actual_query", {"args": [
            "run", "--table", "transactions",
            "--select", "id,date,amount,payee.name,imported_payee,notes,account,account.name,is_child,parent_id,transfer_id,cleared,reconciled",
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
                        "account": c.get("account"),
                        "account.name": c.get("account.name"),
                        "cleared": c.get("cleared"),
                        "reconciled": c.get("reconciled"),
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
                                                "account": tx.get("account", c.get("account")),
                                                "account.name": acc,
                                                "cleared": tx.get("cleared", c.get("cleared")),
                                                "reconciled": tx.get("reconciled", c.get("reconciled"))
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
        receipt_cents_for_actual = int(round(-amount * 100))
        
        # Resolve account ID
        account_key = parsed.get("account_key")
        receipt_account_id = None
        if account_key:
            mapping_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../actual-utils/cache/account-mapping.json"))
            if os.path.exists(mapping_path):
                try:
                    with open(mapping_path, "r") as f:
                        mapping = json.load(f)
                    receipt_account_id = mapping.get(account_key)
                except Exception:
                    pass
                    
        # Extract transfer account
        proposed_txs = state.get("proposed_transactions") or []
        receipt_transfer_account = None
        if len(proposed_txs) == 1:
            receipt_transfer_account = proposed_txs[0].get("transfer_account")
        
        session = await ActualMCPClient.get_session()
        
        candidates = await get_actual_candidates(session, date)
        if not candidates:
            print("  No candidates found in date range, falling through to agent...")
            return {"actual_tx_id": "unknown_id", "status": "fallback_to_agent"}
            
        # Score and filter
        scored = []
        for c in candidates:
            c["match_score"] = score_candidate(payee, receipt_cents_for_actual, date, receipt_account_id, receipt_transfer_account, c)
            scored.append(c)
            
        scored.sort(key=lambda x: x["match_score"], reverse=True)
        
        if not scored:
            print("  All candidates already tagged or none valid, falling through to agent...")
            return {"actual_tx_id": "unknown_id", "status": "fallback_to_agent"}
            
        top = scored[0]
        top_score = top["match_score"]
        final_id = "unknown_id"
        
        if top_score >= SCORE_AUTO_ACCEPT:
            final_id = top["id"]
            print(f"✅ Auto-matched (score {top_score:.3f}): {final_id}")
            print_matched_transaction(top)
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
        import traceback
        return {"status": "error", "error": f"Deterministic search failed: {str(e)}", "traceback": traceback.format_exc()}


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
        
        # Pre-fetch existing receipts from PocketBase for candidates that have #receipt
        candidate_ids_with_receipts = [c.get("id") for c in clean_candidates if c.get("id") and "#receipt" in (c.get("notes") or "")]
        previously_attached_receipts_map = {}
        if candidate_ids_with_receipts:
            try:
                from actual_orch.clients.pocketbase import get_pb_client
                pb_client = get_pb_client()
                for cid in candidate_ids_with_receipts:
                    try:
                        records = pb_client.collection("receipts").get_list(1, 10, query_params={
                            "filter": f"tx_id='{cid}'",
                            "fields": "payee,amount,date"
                        })
                        receipts = [{"payee": getattr(r, "payee", ""), "amount": getattr(r, "amount", 0.0), "date": str(getattr(r, "date", ""))[:10]} for r in records.items]
                        if receipts:
                            previously_attached_receipts_map[cid] = receipts
                    except Exception as e:
                        print(f"  ⚠️ Warning: Failed to fetch receipts for candidate {cid}: {e}")
            except Exception as e:
                print(f"  ⚠️ Warning: Could not import or use pocketbase client: {e}")
        
        narrowed_candidates = []
        entries_by_id = {}
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
                "already_attached_receipts": previously_attached_receipts_map.get(c.get("id"), []) if "#receipt" in (c.get("notes") or "") else [],
                "allowed_to_update": not (c.get("cleared") or c.get("reconciled"))
            }
                
            if c.get("is_child"):
                entry["is_child"] = True
                entry["parent_id"] = c.get("parent_id")
                
            if c.get("transfer_id"):
                entry["transfer_id"] = c.get("transfer_id")
                
            entries_by_id[entry["id"]] = entry
            
        structured_candidates = []
        processed_ids = set()
        
        for c in clean_candidates:
            cid = c.get("id")
            if cid in processed_ids:
                continue
                
            entry = entries_by_id[cid]
            
            # 1. Handle Transfers
            if entry.get("transfer_id"):
                tid = entry["transfer_id"]
                transfer_txs = [entries_by_id[k] for k, v in entries_by_id.items() if v.get("transfer_id") == tid]
                if len(transfer_txs) > 1:
                    structured = {
                        "transfer_id": tid,
                        "type": "transfer",
                        "date": transfer_txs[0]["date"],
                        "transactions": []
                    }
                    for tx in transfer_txs:
                        tx_copy = dict(tx)
                        tx_copy.pop("transfer_id", None)
                        tx_copy.pop("is_child", None)
                        tx_copy.pop("parent_id", None)
                        structured["transactions"].append(tx_copy)
                        processed_ids.add(tx["id"])
                    structured_candidates.append(structured)
                    continue
                    
            # 2. Handle Splits (Parent)
            children = [entries_by_id[k] for k, v in entries_by_id.items() if v.get("is_child") and v.get("parent_id") == cid]
            if children:
                structured = {
                    "id": cid,
                    "type": "split",
                    "date": entry["date"],
                    "amount": entry["amount"],
                    "payee": entry["payee"],
                    "account": entry["account"],
                    "notes": entry["notes"],
                    "already_attached_receipts": entry["already_attached_receipts"],
                    "subtransactions": []
                }
                for child in children:
                    child_copy = dict(child)
                    child_copy.pop("is_child", None)
                    child_copy.pop("parent_id", None)
                    structured["subtransactions"].append(child_copy)
                    processed_ids.add(child["id"])
                structured_candidates.append(structured)
                processed_ids.add(cid)
                continue
                
            # 3. Handle standalone children (e.g. 0-amount parent)
            if entry.get("is_child"):
                pid = entry["parent_id"]
                siblings = [entries_by_id[k] for k, v in entries_by_id.items() if v.get("is_child") and v.get("parent_id") == pid]
                if len(siblings) > 1:
                    structured = {
                        "id": pid,
                        "type": "split",
                        "date": entry["date"],
                        "amount": 0.0,
                        "payee": entry["payee"],
                        "account": entry["account"],
                        "notes": entry["notes"],
                        "already_attached_receipts": [],
                        "subtransactions": []
                    }
                    for sib in siblings:
                        sib_copy = dict(sib)
                        sib_copy.pop("is_child", None)
                        sib_copy.pop("parent_id", None)
                        structured["subtransactions"].append(sib_copy)
                        processed_ids.add(sib["id"])
                    structured_candidates.append(structured)
                    continue

            # 4. Standalone normal transaction
            entry_copy = dict(entry)
            entry_copy.pop("is_child", None)
            entry_copy.pop("parent_id", None)
            entry_copy.pop("transfer_id", None)
            structured_candidates.append(entry_copy)
            processed_ids.add(cid)
            
        def get_sort_key(c):
            try:
                cd = datetime.strptime(c.get("date"), "%Y-%m-%d")
                rd = datetime.strptime(date, "%Y-%m-%d")
                dd = abs((cd - rd).days)
            except Exception:
                dd = 999
            
            amounts_to_check = []
            if "amount" in c:
                amounts_to_check.append(abs(c["amount"]))
            if c.get("type") == "split":
                for sub in c.get("subtransactions", []):
                    amounts_to_check.append(abs(sub["amount"]))
            if c.get("type") == "transfer":
                for tx in c.get("transactions", []):
                    amounts_to_check.append(abs(tx["amount"]))
                    
            min_a_diff = min([abs((amt * 100) - receipt_cents) for amt in amounts_to_check]) if amounts_to_check else 9999999
            return (dd, min_a_diff)
                
        structured_candidates.sort(key=get_sort_key)
        structured_candidates = structured_candidates[:50]
        
        parse_attempts = state.get("parse_attempts", [])
        already_retried = any(a.get("model") == "search_agent" for a in parse_attempts)
        
        if not structured_candidates:
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
        
        # Prepare receipt details for the prompt
        parsed_prompt = dict(parsed)
        parsed_prompt.pop("proposed_transactions", None)
        
        account_key = parsed_prompt.pop("account_key", None)
        if account_key:
            try:
                mapping_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../actual-utils/cache/account-mapping.json"))
                if os.path.exists(mapping_path):
                    with open(mapping_path, "r") as f:
                        mapping = json.load(f)
                    acc_id = mapping.get(account_key)
                    if acc_id:
                        res = await session.call_tool("actual_accounts", {"args": ["list", "--format", "json"]})
                        if res.content:
                            accounts = json.loads(res.content[0].text)
                            acc_match = next((a for a in accounts if a.get("id") == acc_id), None)
                            if acc_match:
                                parsed_prompt["account"] = strip_emojis(acc_match.get("name"))
                            else:
                                parsed_prompt["account"] = strip_emojis(account_key)
                    else:
                        parsed_prompt["account"] = strip_emojis(account_key)
                else:
                    parsed_prompt["account"] = strip_emojis(account_key)
            except Exception as e:
                print(f"  ⚠️ Failed to translate account_key: {e}")
                parsed_prompt["account"] = strip_emojis(account_key)

        if state.get("source") == "email" and state.get("email_context"):
            parsed_prompt["email_context"] = state.get("email_context")
            
        try:
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../actual-utils/config.json"))
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                if config_data.get("AGENT_USER_CONTEXT"):
                    parsed_prompt["AGENT_USER_CONTEXT"] = config_data["AGENT_USER_CONTEXT"]
        except Exception as e:
            print(f"  ⚠️ Failed to load AGENT_USER_CONTEXT: {e}")

        prompt_text = f"""You are a financial matching assistant.

Receipt details:
{json.dumps(parsed_prompt, indent=2)}

Candidate transactions from the bank:
{json.dumps(structured_candidates, indent=2)}

Task: Identify the transaction that matches this receipt. You must respond in JSON.
Rules:
- Amounts may differ due to tip, currency difference, etc. Make sure there is a good reason for the difference.
- Split transactions (type='split') contain 'subtransactions'. Match on the individual subtransaction 'id' (not the parent 'id') if the receipt corresponds to a specific portion of the split.
- Transfers (type='transfer') represent money moving between accounts and contain a 'transactions' array with both sides of the transfer. Match on the specific transaction 'id' within the array that corresponds to the account the receipt is for.
- Payee names might differ but should be conceptually similar, but transfers will often have completely different payee names. Sometimes the notes might contain payee name too (often in a # tag)
- Date should be within 1 day. However, often rent income and bill payments are shifted to the start of the next month.
- Candidates have an 'already_attached_receipts' array. If it is not empty, it contains details (payee, amount, date) of the receipt(s) already attached to this candidate. Only select such a candidate if you are absolutely sure the current receipt is a different version or additional document for the exact same transaction.
- Each candidate has an 'allowed_to_update' flag. If 'allowed_to_update' is true, you can match the transaction and update the payee, transfer_account, notes, or amount, but ONLY if you are absolutely sure it's the same transaction and the current receipt has new info (such as tip amount, or transfer account which often happens with self e-transfers). Use the updated_* fields in your JSON response for this. If updating the amount, the polarity must match the receipt (positive for expense, negative for deposit). If 'allowed_to_update' is false, you MUST NOT provide any updated_* fields.

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
                # Validate case-insensitively and use the exact casing from the candidate list
                matched = next((c for c in clean_candidates if c["id"].lower() == final_id.lower()), None)
                if matched:
                    final_id = matched["id"]
                    print(f"✅ Agent selected valid candidate: {final_id}")
                    print_matched_transaction(matched)
                    
                    # Validate amount (clean_candidates has amount in cents)
                    actual_amt = abs(matched.get("amount", 0))
                    min_allowed = receipt_cents * 0.90
                    max_allowed = receipt_cents * 1.30
                    if getattr(result, "updated_amount", None) is not None:
                        print(f"  Agent updating amount to ${result.updated_amount:.2f} (original: ${actual_amt/100:.2f})")
                    elif not (min_allowed <= actual_amt <= max_allowed):
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

        update_transaction = {}
        if final_id != "unknown_id" and result:
            if getattr(result, "updated_amount", None) is not None:
                update_transaction["amount"] = result.updated_amount
            if getattr(result, "updated_payee", None) is not None:
                update_transaction["payee"] = result.updated_payee
            if getattr(result, "updated_notes", None) is not None:
                update_transaction["notes"] = result.updated_notes
            if getattr(result, "updated_transfer_account", None) is not None:
                update_transaction["transfer_account"] = result.updated_transfer_account

        return {"actual_tx_id": final_id, "actual_tx_id_transfer": transfer_id, "update_transaction": update_transaction}
        
    except Exception as e:
        import traceback
        return {"status": "error", "error": f"Agent search failed: {str(e)}", "traceback": traceback.format_exc()}

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
            
        # Fetch accounts and payees for transfer resolution
        try:
            acct_res = await session.call_tool("actual_accounts", {"args": ["list", "--format", "json"]})
            payee_res = await session.call_tool("actual_payees", {"args": ["list", "--format", "json"]})
            all_accounts = json.loads(acct_res.content[0].text) if acct_res.content else []
            all_payees = json.loads(payee_res.content[0].text) if payee_res.content else []
        except Exception as e:
            print(f"  ⚠️ Failed to fetch accounts or payees for transfer resolution: {e}")
            all_accounts = []
            all_payees = []

        import re
        def clean_acct(s):
            return re.sub(r'[\u2600-\u27BF\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2300-\u23FF\u2B00-\u2BFF\uFE00-\uFE0F]+', '', s).strip()

        def get_transfer_payee(transfer_name):
            if not transfer_name:
                return None
            clean_transfer = clean_acct(transfer_name)
            for acc in all_accounts:
                if clean_acct(acc.get("name", "")) == clean_transfer:
                    acc_id = acc.get("id")
                    for p in all_payees:
                        if p.get("transfer_acct") == acc_id:
                            return p.get("id")
            return None
        
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
                
                transfer_account = pt.get("transfer_account")
                transfer_payee_id = get_transfer_payee(transfer_account)
                
                if transfer_payee_id:
                    sub["payee"] = transfer_payee_id
                else:
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
            # Single transaction — check for transfer or category
            if proposed_txs:
                pt = proposed_txs[0]
                transfer_account = pt.get("transfer_account")
                transfer_payee_id = get_transfer_payee(transfer_account)
                
                if transfer_payee_id:
                    # It's a transfer, so override payee_name with payee ID
                    del new_tx["payee_name"]
                    new_tx["payee"] = transfer_payee_id
                else:
                    pt_cat = pt.get("category")
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
                "--run-transfers",
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
                "--select", "id,amount,payee.name,notes,is_child,parent_id,transfer_id",
                "--format", "json"
            ]})
            if q_res.content and len(q_res.content) > 0:
                txs = json.loads(q_res.content[0].text)
                if txs and len(txs) > 0:
                    # 1. Try to find the exact parent transaction match (for single/normal tx)
                    for tx in txs:
                        if not tx.get("is_child") and tx.get("amount") == expense_cents and tx.get("notes") and "#receipt" in tx["notes"]:
                            print(f"  ✅ Retrieved new transaction ID: {tx['id']}")
                            return {"actual_tx_id": tx["id"], "actual_tx_id_transfer": tx.get("transfer_id"), "status": "continue"}
                            
                    # 2. Try to find a child transaction match (for split tx)
                    for tx in txs:
                        if tx.get("is_child") and tx.get("parent_id"):
                            # Match if payee is the same and amount matches one of the subtransactions
                            sub_amounts = [int(round(-pt.get("amount", 0.0) * 100)) for pt in proposed_txs]
                            if (payee and tx.get("payee.name") == payee) or (tx.get("amount") in sub_amounts):
                                print(f"  ✅ Retrieved new split transaction parent ID: {tx['parent_id']}")
                                # A split sub-transaction might be a transfer, but Actual maps transfer_id on the child. We need to query parent or just pass child transfer_id if present.
                                # Since Pocketbase maps to parent, we'll try to find any transfer_id in siblings.
                                transfer_id = None
                                for sibling in txs:
                                    if sibling.get("parent_id") == tx["parent_id"] and sibling.get("transfer_id"):
                                        transfer_id = sibling.get("transfer_id")
                                        break
                                return {"actual_tx_id": tx["parent_id"], "actual_tx_id_transfer": transfer_id, "status": "continue"}
                                
                    # 3. Fallback to first returned transaction
                    fallback_tx = txs[0]
                    fallback_id = fallback_tx.get("parent_id") if fallback_tx.get("is_child") else fallback_tx["id"]
                    transfer_id = fallback_tx.get("transfer_id")
                    if fallback_tx.get("is_child"):
                        for sibling in txs:
                            if sibling.get("parent_id") == fallback_id and sibling.get("transfer_id"):
                                transfer_id = sibling.get("transfer_id")
                                break
                    print(f"  ✅ Retrieved new transaction ID (fallback): {fallback_id}")
                    return {"actual_tx_id": fallback_id, "actual_tx_id_transfer": transfer_id, "status": "continue"}
            
            print("  ⚠️ Created transaction but couldn't query its ID. PocketBase won't have the ID.")
            return {"actual_tx_id": "unknown_id", "status": "continue"}
            
    except Exception as e:
        import traceback
        return {"status": "error", "error": f"Auto-create failed: {str(e)}", "traceback": traceback.format_exc()}
    return {}

async def update_actual_transaction_node(state: dict):
    """Update matched/created transaction with #receipt and any agent updates in Actual Budget."""
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
                "--select", "id,date,amount,payee,payee.name,imported_payee,notes,category,account,account.name,cleared,reconciled,is_child,parent_id,sort_order,transfer_id,schedule,imported_id",
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
        
        update_transaction = state.get("update_transaction") or {}

        # Summarise what we know about the matched transaction before proceeding
        if tx_data:
            print_matched_transaction(tx_data, existing_notes)

        if update_transaction:
            fields = {k: v for k, v in update_transaction.items() if k != "notes"}
            if fields:
                parts = []
                for k, v in fields.items():
                    if k == "amount":
                        parts.append(f"amount → ${v:.2f}")
                    else:
                        parts.append(f"{k} → {v}")
                print(f"  ✏️  Agent updates: {', '.join(parts)}")
        
        # Determine base notes and append #receipt
        base_notes = update_transaction.get("notes", existing_notes)
        if "#receipt" not in base_notes:
            new_notes = f"{base_notes.strip()} #receipt".strip()
        else:
            new_notes = base_notes
            
        update_transaction["notes"] = new_notes

        # Build update payload with only writable fields to prevent the update API
        # from choking on relational/read-only fields returned by actual_query
        # (e.g. payee.name, account.name, is_child, parent_id, transfer_id).
        # This also preserves existing child transaction fields that would otherwise
        # be wiped out if we only sent {"notes": ...}.
        WRITABLE_FIELDS = {"date", "amount", "payee", "payee_name", "imported_payee",
                           "notes", "category", "account", "cleared", "imported_id", "sort_order"}
        update_payload = {}
        if tx_data:
            update_payload = {k: v for k, v in tx_data.items() if k in WRITABLE_FIELDS}
            
        update_payload["notes"] = update_transaction["notes"]

        if tx_data and not tx_data.get("cleared") and not tx_data.get("reconciled"):
            if "amount" in update_transaction:
                update_amount = update_transaction["amount"]
                update_cents = int(round(-update_amount * 100))
                update_payload["amount"] = update_cents
                print(f"Agent requested amount update to: {update_cents}")
            
            proposed_txs = state.get("parsed_data", {}).get("proposed_transactions", [])
            parsed_payee = state.get("parsed_data", {}).get("payee")

            target_transfer_account = update_transaction.get("transfer_account")
            if not target_transfer_account and len(proposed_txs) == 1:
                target_transfer_account = proposed_txs[0].get("transfer_account")

            target_payee_name = update_transaction.get("payee") or parsed_payee

            if target_transfer_account:
                transfer_acc_name = target_transfer_account
                try:
                    acct_res = await session.call_tool("actual_accounts", {"args": ["list", "--format", "json"]})
                    payee_res = await session.call_tool("actual_payees", {"args": ["list", "--format", "json"]})
                    all_accounts = json.loads(acct_res.content[0].text) if acct_res.content else []
                    all_payees = json.loads(payee_res.content[0].text) if payee_res.content else []
                    
                    def clean_acct_name(s):
                        return re.sub(r'[\u2600-\u27BF\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2300-\u23FF\u2B00-\u2BFF\uFE00-\uFE0F]+', '', s).strip()
                        
                    clean_transfer = clean_acct_name(transfer_acc_name)
                    transfer_payee_id = None
                    for acc in all_accounts:
                        if clean_acct_name(acc.get("name", "")) == clean_transfer:
                            acc_id = acc.get("id")
                            for p in all_payees:
                                if p.get("transfer_acct") == acc_id:
                                    transfer_payee_id = p.get("id")
                                    break
                            break
                            
                    if transfer_payee_id:
                        update_payload["payee"] = transfer_payee_id
                        if "payee_name" in update_payload:
                            del update_payload["payee_name"]
                        print(f"Agent requested transfer update to: {transfer_acc_name} ({transfer_payee_id})")
                    elif target_payee_name:
                        update_payload["payee_name"] = target_payee_name
                        if "payee" in update_payload:
                            del update_payload["payee"]
                        print(f"Agent requested payee update to: {target_payee_name} (Transfer account fallback)")
                except Exception as e:
                    print(f"⚠️ Failed to resolve transfer account {transfer_acc_name}: {e}")
                    if target_payee_name:
                        update_payload["payee_name"] = target_payee_name
                        if "payee" in update_payload:
                            del update_payload["payee"]
                        print(f"Agent requested payee update to: {target_payee_name} (Transfer account fallback)")
            elif target_payee_name:
                update_payload["payee_name"] = target_payee_name
                if "payee" in update_payload:
                    del update_payload["payee"]
                print(f"Agent requested payee update to: {target_payee_name}")

        proposed_txs = state.get("parsed_data", {}).get("proposed_transactions", [])
        if tx_data and not tx_data.get("cleared") and not tx_data.get("reconciled") and not tx_data.get("is_child") and len(proposed_txs) > 1:
            print("  ➕ Transforming matched standalone transaction into a split transaction...")
            update_payload["subtransactions"] = []
            
            cats = []
            try:
                import os
                categories_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../actual-utils/cache/categories.json"))
                if os.path.exists(categories_path):
                    with open(categories_path, "r") as f:
                        loaded_cats = json.load(f)
                        if isinstance(loaded_cats, list):
                            cats = loaded_cats
            except Exception:
                pass

            try:
                acct_res = await session.call_tool("actual_accounts", {"args": ["list", "--format", "json"]})
                payee_res = await session.call_tool("actual_payees", {"args": ["list", "--format", "json"]})
                all_accounts = json.loads(acct_res.content[0].text) if acct_res.content else []
                all_payees = json.loads(payee_res.content[0].text) if payee_res.content else []
            except Exception as e:
                print(f"  ⚠️ Failed to fetch accounts or payees for split resolution: {e}")
                all_accounts = []
                all_payees = []

            def get_transfer_payee_update(transfer_name):
                if not transfer_name:
                    return None
                clean_transfer = strip_emojis(transfer_name)
                for acc in all_accounts:
                    if strip_emojis(acc.get("name", "")) == clean_transfer:
                        acc_id = acc.get("id")
                        for p in all_payees:
                            if p.get("transfer_acct") == acc_id:
                                return p.get("id")
                return None

            for pt in proposed_txs:
                sub_cents = int(round(-pt.get("amount", 0.0) * 100))
                sub = {
                    "amount": sub_cents,
                    "notes": pt.get("notes", "")
                }
                
                transfer_account = pt.get("transfer_account")
                transfer_payee_id = get_transfer_payee_update(transfer_account)
                
                if transfer_payee_id:
                    sub["payee"] = transfer_payee_id
                else:
                    pt_cat = pt.get("category")
                    if pt_cat:
                        cat_match = next((c for c in cats if c.get("name") == pt_cat or c.get("id") == pt_cat), None)
                        sub["category"] = cat_match.get("id") if cat_match else pt_cat
                        
                update_payload["subtransactions"].append(sub)
                
            # Parent of a split should not have a category
            if "category" in update_payload:
                update_payload["category"] = None

            try:
                # Delete old standalone transaction
                await session.call_tool("actual_transactions", {"args": ["delete", tx_id]})
                
                # Add new split transaction
                add_res = await session.call_tool("actual_transactions", {"args": [
                    "add", "--account", tx_data["account"],
                    "--data", json.dumps([update_payload])
                ]})
                
                try:
                    added_ids = json.loads(add_res.content[0].text)
                    if isinstance(added_ids, list) and len(added_ids) > 0:
                        new_tx_id = added_ids[0]
                        print(f"✅ Successfully converted transaction into a split (Old ID: {tx_id}, New ID: {new_tx_id})")
                        return {"actual_tx_id": new_tx_id, "replaced_tx_id": tx_id}
                except Exception as e:
                    print(f"⚠️ Failed to parse new split transaction ID: {e}")
            except Exception as e:
                print(f"⚠️ Failed to recreate split transaction: {e}")
            return {}

        mcp_tools = await load_mcp_tools(session)
        tx_tool = next((t for t in mcp_tools if t.name == "actual_transactions"), None)
        if tx_tool:
            await tx_tool.ainvoke({"args": [
                "update", tx_id,
                "--data", json.dumps(update_payload)
            ]})
            print(f"✅ Successfully updated transaction {tx_id} (Tagged with #receipt)")
            
    except Exception as e:
        print(f"⚠️ Failed to tag transaction (non-fatal): {e}")
    return {}
