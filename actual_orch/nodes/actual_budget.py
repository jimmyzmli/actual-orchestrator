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
# Deterministic search: query + score (no agent)
# ---------------------------------------------------------------------------

async def _deterministic_search(session, receipt_date: str, receipt_cents: int,
                                 receipt_payee: str) -> list[dict]:
    """
    Query Actual Budget for candidate transactions within ±7 days of
    *receipt_date*, score them, and return them sorted best-first.
    Each returned dict has an extra 'match_score' key.
    """
    try:
        rd = datetime.strptime(receipt_date, "%Y-%m-%d")
    except ValueError:
        return []

    start_date = (rd - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date   = (rd + timedelta(days=7)).strftime("%Y-%m-%d")

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
            if c.get("transfer_id"):
                continue  # Skip transfers
                
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
                        "account.name": c.get("account.name")
                    }
                parent_map[pid]["amount"] += c.get("amount", 0)
                # Ensure we catch #receipt if it's on any child
                if "#receipt" in (c.get("notes") or "") and "#receipt" not in parent_map[pid]["notes"]:
                    parent_map[pid]["notes"] += " #receipt"
            else:
                candidates.append(c)
        candidates.extend(parent_map.values())
        
    except Exception as e:
        print(f"⚠️ Deterministic query failed: {e}")
        return []

    # Score every candidate
    scored = []
    for c in candidates:
        # Skip transactions that already have #receipt tag (already matched)
        notes = c.get("notes") or ""
        if "#receipt" in notes:
            continue
        c["match_score"] = score_candidate(receipt_payee, receipt_cents, receipt_date, c)
        scored.append(c)

    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# LLM tiebreaker for ambiguous matches
# ---------------------------------------------------------------------------

async def _llm_tiebreaker(candidates: list[dict], receipt_payee: str,
                           receipt_cents: int, receipt_date: str) -> Optional[str]:
    """
    Ask an LLM to pick the best candidate from a short-list.
    Returns the transaction UUID or None if no good match.
    """
    if not candidates:
        return None

    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com/v1",
        max_tokens=256,
    )

    candidate_descriptions = []
    for i, c in enumerate(candidates[:5]):  # Max 5 candidates
        candidate_descriptions.append(
            f"  {i+1}. ID: {c['id']}\n"
            f"     Date: {c.get('date', '?')}\n"
            f"     Amount: ${abs(c.get('amount', 0))/100:.2f}\n"
            f"     Payee: {c.get('payee.name', '?')}\n"
            f"     Imported Payee: {c.get('imported_payee', '?')}\n"
            f"     Account: {c.get('account.name', '?')}\n"
            f"     Score: {c.get('match_score', 0):.3f}"
        )

    prompt = f"""You are a financial matching assistant. A receipt has been scanned and we need to find the matching bank transaction.

Receipt details:
- Date: {receipt_date}
- Amount: ${receipt_cents/100:.2f}
- Payee: {receipt_payee}

Candidate transactions from the bank:
{chr(10).join(candidate_descriptions)}

Task: Pick the BEST matching transaction, or say NONE if none of them are a plausible match for this receipt.

Rules:
- The payee name on the receipt may differ from the bank's payee name (e.g., "Walmart" vs "WAL-MART SUPERCENTER #1234"). Use your judgment.
- Small amount differences (tips, rounding, tax adjustments) are acceptable.
- If the amounts are very different (>30% off), it's probably not a match unless the payees are a very strong match.
- If there are multiple good candidates, pick the one with the closest amount AND most similar payee.

YOUR RESPONSE MUST BE EXACTLY ONE OF:
- The 36-character UUID of the best match (e.g., "abc12345-1234-1234-1234-123456789abc")
- The word "NONE" if no candidate is a good match

Do not explain your reasoning. Just output the UUID or NONE."""

    print(f"  [Token Est] Tiebreaker prompt: ~{estimate_tokens(prompt)} tokens")

    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        content = result.content.strip()

        import re
        uuid_match = re.search(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            content, re.IGNORECASE,
        )
        if uuid_match:
            matched_id = uuid_match.group(0)
            # Validate that this ID is actually in our candidate list
            valid_ids = {c["id"] for c in candidates}
            if matched_id in valid_ids:
                return matched_id
            else:
                print(f"⚠️ LLM returned UUID {matched_id} not in candidate list, ignoring.")
        return None
    except Exception as e:
        print(f"⚠️ LLM tiebreaker failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main search node (hybrid: deterministic → LLM tiebreaker → agent fallback)
# ---------------------------------------------------------------------------

SCORE_AUTO_ACCEPT = 0.70
SCORE_LLM_TIEBREAKER = 0.40

async def search_actual_transaction_node(state: dict):
    """Search for the transaction in Actual Budget using a hybrid approach:
    1. Deterministic query + scoring
    2. LLM tiebreaker for ambiguous matches
    3. Agent fallback for when nothing is found
    """
    final_id = "unknown_id"
    
    try:
        print("Matching transaction in Actual Budget...")
        parsed = state.get("parsed_data")
        if not parsed:
            return {"status": "error", "error": "No parsed data found"}
            
        date = parsed["date"]
        amount = parsed["total"]
        payee = parsed["payee"]
        receipt_cents = abs(int(amount * 100))
        
        session = await ActualMCPClient.get_session()
        
        # ── Step 1: Deterministic search ──
        scored = await _deterministic_search(session, date, receipt_cents, payee)
        
        if scored:
            top = scored[0]
            top_score = top["match_score"]
            print(f"  Top candidate: {top['id']} "
                  f"(score={top_score:.3f}, payee={top.get('payee.name', '?')}, "
                  f"amount=${abs(top.get('amount', 0))/100:.2f})")
            
            if top_score >= SCORE_AUTO_ACCEPT:
                # High confidence — auto-accept
                final_id = top["id"]
                print(f"✅ Auto-matched (score {top_score:.3f}): {final_id}")
            
            elif top_score >= SCORE_LLM_TIEBREAKER:
                # Ambiguous — ask LLM to pick
                # Send candidates scoring above threshold
                ambiguous = [c for c in scored if c["match_score"] >= SCORE_LLM_TIEBREAKER]
                print(f"  🤔 Ambiguous ({len(ambiguous)} candidates ≥ {SCORE_LLM_TIEBREAKER}), asking LLM to pick...")
                picked = await _llm_tiebreaker(ambiguous, payee, receipt_cents, date)
                if picked:
                    final_id = picked
                    picked_score = next((c["match_score"] for c in ambiguous if c["id"] == picked), 0)
                    print(f"✅ LLM picked (score {picked_score:.3f}): {final_id}")
                else:
                    print(f"  LLM said NONE, falling through to agent...")
            else:
                print(f"  No candidate scored ≥ {SCORE_LLM_TIEBREAKER}, falling through to agent...")
        else:
            print(f"  No candidates found in date range, falling through to agent...")

        # ── Step 2: Validate the deterministic match ──
        if final_id != "unknown_id":
            # Validate amount is within tolerance
            matched_candidate = next((c for c in scored if c["id"] == final_id), None)
            if matched_candidate:
                actual_amt = abs(matched_candidate.get("amount", 0))
                min_allowed = receipt_cents * 0.90
                max_allowed = receipt_cents * 1.30
                if not (min_allowed <= actual_amt <= max_allowed):
                    print(f"❌ VALIDATION FAILED: Bank amount ${actual_amt/100:.2f} "
                          f"outside 0.90x–1.30x of receipt ${amount:.2f}. Rejecting.")
                    final_id = "unknown_id"

        # ── Step 3: Agent fallback ──
        if final_id == "unknown_id":
            print("  🔄 Falling back to LLM structured selection...")
            
            narrowed_candidates = []
            for c in scored:
                narrowed_candidates.append({
                    "id": c.get("id"),
                    "date": c.get("date"),
                    "amount": c.get("amount"),
                    "payee": c.get("payee.name") or c.get("imported_payee"),
                    "account": c.get("account.name"),
                    "notes": (c.get("notes") or "")[:200]
                })
            # Limit to top 10 candidates to save tokens
            narrowed_candidates = narrowed_candidates[:10]
            
            llm = ChatOpenAI(
                model="deepseek-v4-flash", 
                api_key=os.environ.get("DEEPSEEK_API_KEY", ""), 
                base_url="https://api.deepseek.com/v1"
            )
            
            structured_llm = llm.with_structured_output(TransactionMatchResult, method="json_mode")
            
            prompt_text = f"""You are a financial assistant connecting to Actual Budget.
We have a new parsed receipt:
- Date: {date}
- Amount: ${amount}
- Payee: {payee}

Here are the top candidate transactions from the bank (within +/- 7 days):
{json.dumps(narrowed_candidates, indent=2) if narrowed_candidates else "[]"}

Task:
1. Review the provided candidate transactions. If one is a good match based on Payee and Amount similarity, set `matched_id` to its ID.
2. Note that the receipt amount and bank amount may differ significantly due to foreign currency conversion or added tips. If the Payee and Date are a very strong match, it is highly likely to be the correct transaction even if the amount differs.
3. If NO good match is found in the candidates, you must return null for `matched_id`.
4. Output EXACTLY a JSON object matching this schema:
{{
  "matched_id": "string UUID or null"
}}"""
            
            print(f"  [Token Est] Fallback agent prompt: ~{estimate_tokens(prompt_text)} tokens")
            
            from actual_orch.shared.peak_hours import pause_if_peak_hours
            pause_if_peak_hours("deepseek")
            
            try:
                result = await structured_llm.ainvoke([HumanMessage(content=prompt_text)])
                if result.matched_id:
                    final_id = result.matched_id
                    print(f"Agent matched tx: {final_id}")
                    # Validate that the ID actually exists in the narrowed candidates
                    matched = next((c for c in narrowed_candidates if c["id"] == final_id), None)
                    if matched:
                        print(f"✅ Agent selected valid candidate: {final_id}")
                    else:
                        print(f"❌ VALIDATION FAILED: Agent claimed to find tx {final_id}, but it was not in the narrowed candidates.")
                        final_id = "unknown_id"
                else:
                    print(f"Agent output didn't contain matched_id")
            except Exception as e:
                print(f"Warning: Failed to parse structured output from LLM: {e}")
                
    except Exception as e:
        if final_id != "unknown_id" and ("TaskGroup" in str(e) or "JSONRPCMessage" in str(e) or "BrokenResourceError" in str(e)):
            print("Ignored MCP shutdown error, agent succeeded.")
        else:
            return {"status": "error", "error": f"Actual Budget search failed: {str(e)}"}
            
    transfer_id = None
    if final_id != "unknown_id":
        matched = next((c for c in scored if c["id"] == final_id), None)
        if matched and matched.get("transfer_id"):
            transfer_id = matched.get("transfer_id")
            
    return {"actual_tx_id": final_id, "actual_tx_id_transfer": transfer_id}


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
