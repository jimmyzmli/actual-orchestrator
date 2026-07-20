import os
import json
import base64
import difflib
from datetime import datetime, timedelta
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from pdf2image import convert_from_path

from tools.actual_mcp import find_or_create_transaction
from tools.pocketbase_client import save_receipt, check_receipt_exists
from graph.cache import get_ocr, set_ocr, get_parsed, set_parsed, _file_hash

class ReceiptState(TypedDict, total=False):
    file_path: str        # Local path to the downloaded PDF
    gdrive_path: str      # Original gdrive path (for source_ref)
    ocr_text: str         # Text extracted by Gemini
    parsed_data: Dict[str, Any]
    actual_tx_id: str
    status: str
    error: str
    image_paths: List[str] # Paths to extracted JPEG images
    proposed_transaction: str # JSON string of proposed transaction if not found
    human_approved: bool

# Pydantic model for Deepseek parsing
class ReceiptItem(BaseModel):
    description: str = Field(description="Description of the item")
    amount: float = Field(description="Amount of the item")

class ParsedReceipt(BaseModel):
    date: str = Field(description="Date of the receipt in YYYY-MM-DD format")
    total: float = Field(description="Total amount of the receipt")
    payee: str = Field(description="Name of the store or payee")
    items: List[ReceiptItem] = Field(description="List of individual items on the receipt")

def skip_if_processed(state: ReceiptState):
    """Check PocketBase source_hash to skip already-processed files."""
    file_path = state.get("file_path", "")
    if file_path:
        file_hash = _file_hash(file_path)
        if check_receipt_exists(file_hash):
            print(f"⏭️  Already processed in PocketBase (hash {file_hash[:12]}...): {file_path}")
            return {"status": "already_processed"}
    return {}

def extract_ocr_node(state: ReceiptState):
    """Extract raw text from the PDF using Gemini Vision"""
    try:
        file_path = state["file_path"]

        # Always need page images for PocketBase upload
        images = convert_from_path(file_path)
        image_paths = []
        for idx, img in enumerate(images):
            img_path = f"{file_path}_page_{idx}.jpg"
            if not os.path.exists(img_path):
                img.save(img_path, format="JPEG")
            image_paths.append(img_path)

        # Check Redis cache
        cached = get_ocr(file_path)
        if cached is not None:
            print("Using cached OCR text (Redis)...")
            return {"ocr_text": cached, "image_paths": image_paths}

        print(f"Extracting OCR with Gemini Vision for {file_path}...")
        
        # Respect Gemini's 15 Requests Per Minute (RPM) free tier rate limit
        import time
        time.sleep(4.1) 
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            temperature=0
        )
        
        content = [
            {"type": "text", "text": "Extract all the text from this receipt accurately, line by line. Output ONLY the raw text without any markdown formatting or extra commentary."}
        ]
        
        for img_path in image_paths:
            with open(img_path, "rb") as f:
                img_str = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
            })
            
        messages = [
            HumanMessage(content=content)
        ]
        
        result = llm.invoke(messages)
        ocr_text = result.content
        if isinstance(ocr_text, list):
            ocr_text = "".join([block.get("text", "") if isinstance(block, dict) else str(block) for block in ocr_text])
        elif not isinstance(ocr_text, str):
            ocr_text = str(ocr_text)
            
        set_ocr(file_path, ocr_text)
            
        return {"ocr_text": ocr_text, "image_paths": image_paths}
    except Exception as e:
        return {"status": "error", "error": f"OCR failed: {str(e)}"}

def parse_receipt_node(state: ReceiptState):
    """Parse the OCR text using DeepSeek"""
    try:
        if not state.get("ocr_text"):
            return {"status": "error", "error": "No OCR text found"}

        file_path = state["file_path"]

        # Check Redis cache
        cached = get_parsed(file_path)
        if cached is not None:
            print("Using cached parsed data (Redis)...")
            return {"parsed_data": cached}

        print(f"Parsing receipt with DeepSeek...")
        
        # Initialize DeepSeek model
        llm = ChatOpenAI(
            model="deepseek-v4-flash", 
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""), 
            base_url="https://api.deepseek.com/v1",
            max_tokens=1024
        )
        
        structured_llm = llm.with_structured_output(ParsedReceipt, method="json_mode")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at extracting structured data from raw OCR text of receipts. Extract the date, total amount, payee, and line items. If a date is missing, use the current date or guess based on context. Amount should be a number. Date MUST be YYYY-MM-DD. You MUST respond with a JSON object. For the top-level keys, strictly use 'date', 'total', 'payee', and 'items'. For the items array, strictly use the keys 'description' and 'amount'. For the item 'description', write a casual, human-readable description but try to include all useful info (like brand or type) while avoiding weird characters or abbreviations."),
            ("human", "Here is the raw OCR text:\n\n{ocr_text}")
        ])
        
        chain = prompt | structured_llm
        result = chain.invoke({"ocr_text": state["ocr_text"]})
        
        parsed_dict = result.dict()
        set_parsed(file_path, parsed_dict)
            
        return {"parsed_data": parsed_dict}
    except Exception as e:
        return {"status": "error", "error": f"Parsing failed: {str(e)}"}

import asyncio
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent


# ---------------------------------------------------------------------------
# Scoring helpers for deterministic transaction matching
# ---------------------------------------------------------------------------

def _payee_similarity(receipt_payee: str, candidate_payee: str, candidate_imported_payee: str) -> float:
    """
    Fuzzy-match the receipt payee against both the user-facing payee name and the
    raw bank-provided imported_payee.  Returns the best score (0.0–1.0).
    """
    receipt_lower = receipt_payee.lower().strip()
    scores = []
    for name in [candidate_payee, candidate_imported_payee]:
        if not name:
            continue
        name_lower = name.lower().strip()
        # Full SequenceMatcher ratio
        ratio = difflib.SequenceMatcher(None, receipt_lower, name_lower).ratio()
        scores.append(ratio)
        # Also check if one contains the other (common for "WALMART" vs "WALMART SUPERCENTER #1234")
        if receipt_lower in name_lower or name_lower in receipt_lower:
            scores.append(max(ratio, 0.80))
    return max(scores) if scores else 0.0


def _amount_similarity(receipt_cents: int, candidate_cents: int) -> float:
    """
    Score how close the two amounts are.  Both values should be absolute
    (positive) cent values.  Returns 0.0–1.0.
    """
    if receipt_cents == 0 and candidate_cents == 0:
        return 1.0
    if receipt_cents == 0 or candidate_cents == 0:
        return 0.0
    ratio = min(receipt_cents, candidate_cents) / max(receipt_cents, candidate_cents)
    # Exact match bonus
    if receipt_cents == candidate_cents:
        return 1.0
    return ratio


def _date_proximity(receipt_date: str, candidate_date: str, max_days: int = 7) -> float:
    """
    Linear decay over *max_days*.  Same-day → 1.0, max_days away → 0.0.
    """
    try:
        rd = datetime.strptime(receipt_date, "%Y-%m-%d")
        cd = datetime.strptime(candidate_date, "%Y-%m-%d")
        delta = abs((rd - cd).days)
        if delta == 0:
            return 1.0
        if delta >= max_days:
            return 0.0
        return 1.0 - (delta / max_days)
    except (ValueError, TypeError):
        return 0.0


def score_candidate(receipt_payee: str, receipt_cents: int, receipt_date: str,
                     candidate: dict) -> float:
    """
    Return an aggregate match score (0.0–1.0) for a single Actual Budget
    transaction candidate against the parsed receipt.
    
    Weights:
      - Amount similarity:  0.45
      - Payee similarity:   0.40
      - Date proximity:     0.15
    """
    W_AMOUNT = 0.45
    W_PAYEE  = 0.40
    W_DATE   = 0.15

    cand_amount = abs(candidate.get("amount", 0))
    amount_score = _amount_similarity(receipt_cents, cand_amount)

    payee_score = _payee_similarity(
        receipt_payee,
        candidate.get("payee.name") or candidate.get("payee_name", ""),
        candidate.get("imported_payee", ""),
    )

    cand_date = candidate.get("date", "")
    date_score = _date_proximity(receipt_date, cand_date)

    return (W_AMOUNT * amount_score) + (W_PAYEE * payee_score) + (W_DATE * date_score)


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
            "--select", "id,date,amount,payee.name,imported_payee,notes,account.name",
            "--filter", json.dumps(filter_obj),
            "--format", "json",
        ]})
        raw = res.content[0].text if res.content else "[]"
        candidates = json.loads(raw) if isinstance(raw, str) else []
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
                           receipt_cents: int, receipt_date: str) -> str | None:
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

async def search_actual_transaction_node(state: ReceiptState):
    """Search for the transaction in Actual Budget using a hybrid approach:
    1. Deterministic query + scoring
    2. LLM tiebreaker for ambiguous matches
    3. Agent fallback for when nothing is found
    """
    final_id = "unknown_id"
    proposed_tx = None
    
    try:
        print("Matching transaction in Actual Budget...")
        parsed = state.get("parsed_data")
        if not parsed:
            return {"status": "error", "error": "No parsed data found"}
            
        date = parsed["date"]
        amount = parsed["total"]
        payee = parsed["payee"]
        receipt_cents = abs(int(amount * 100))
        
        notes = "Extracted from receipt. Items:\n"
        for item in parsed.get("items", []):
            notes += f"- {item['description']}: ${item['amount']}\n"
            
        server_params = StdioServerParameters(
            command="node",
            args=["wrapper.js"],
            env=os.environ.copy()
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

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
                if final_id == "unknown_id" and proposed_tx is None:
                    print("  🔄 Falling back to agent-based search...")
                    mcp_tools = await load_mcp_tools(session)
                    search_tools = [t for t in mcp_tools if "actual_query" in t.name or "actual_accounts" in t.name]
                    
                    for model_name in ["deepseek-v4-flash", "deepseek-v4-pro"]:
                        if model_name != "deepseek-v4-flash":
                            print(f"Retrying agent match with {model_name}...")
                            
                        llm = ChatOpenAI(
                            model=model_name, 
                            api_key=os.environ.get("DEEPSEEK_API_KEY", ""), 
                            base_url="https://api.deepseek.com/v1"
                        )
                        
                        agent = create_react_agent(llm, tools=search_tools)
                        
                        prompt = f"""
                        You are a financial assistant connecting to Actual Budget.
                        We have a new parsed receipt:
                        - Date: {date}
                        - Amount: ${amount}
                        - Payee: {payee}
                        - Notes to append: {notes}
                        
                        Task:
                        1. Use actual_query to search transactions within +/- 7 days of {date}. Look for an amount between ${amount * 0.90:.2f} and ${amount * 1.30:.2f} (convert to cents, expenses are negative!). Make sure to filter for on budget transactions by including "account.offbudget": false in the filter.
                        2. If found, verify that the payee or transaction context matches '{payee}'. If it's a good match, return its ID. If multiple match, pick the best one based on Payee similarity.
                        3. If NOT found, use actual_accounts to list accounts. Pick the most logical account for this payee (e.g. Chequing or Credit Card, avoid off-budget).
                        
                        IMPORTANT: 
                        - If you FOUND it, your final message MUST be EXACTLY the 36-character TRANSACTION UUID.
                        - If you did NOT find it, your final message MUST be EXACTLY a JSON block containing the proposed transaction: {{"date": "{date}", "amount_cents": <amount in cents>, "payee_name": "{payee}", "account_id": "<selected account id>", "notes": "<notes>"}}. Do not output anything else.
                        """
                        
                        res = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
                        content = res["messages"][-1].content.strip()
                        
                        import re
                        uuid_match = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', content, re.IGNORECASE)
                        if uuid_match:
                            final_id = uuid_match.group(0)
                            print(f"Agent found tx: {final_id}")
                            
                            # FUZZY AMOUNT VALIDATION
                            query_tool = next((t for t in mcp_tools if t.name == "actual_query"), None)
                            if query_tool:
                                try:
                                    q_res = await query_tool.ainvoke({"args": ["run", "--table", "transactions", "--filter", json.dumps({"id": final_id}), "--select", "id,amount", "--format", "json"]})
                                    q_text = q_res[0].get("text", "[]") if isinstance(q_res, list) else getattr(q_res, "text", "[]")
                                    if isinstance(q_text, str):
                                        q_json = json.loads(q_text)
                                        if len(q_json) > 0:
                                            actual_amt = q_json[0].get("amount", 0)
                                            expected_amt = int(amount * 100)
                                            
                                            abs_actual = abs(actual_amt)
                                            abs_expected = abs(expected_amt)
                                            min_allowed = abs_expected * 0.90
                                            max_allowed = abs_expected * 1.30
                                            
                                            if not (min_allowed <= abs_actual <= max_allowed):
                                                print(f"❌ VALIDATION FAILED: Actual Budget amount ${abs_actual/100:.2f} is outside the allowed range for parsed receipt amount ${amount:.2f}. Rejecting match.")
                                                final_id = "unknown_id"
                                except Exception as e:
                                    print(f"Warning: Failed to validate amount: {e}")
                            
                            if final_id != "unknown_id":
                                break  # Success, exit loop
                        else:
                            # Attempt to extract JSON
                            json_match = re.search(r'\{.*\}', content, re.DOTALL)
                            if json_match:
                                proposed_tx = json_match.group(0)
                                print(f"Agent proposed tx: {proposed_tx}")
                            else:
                                print(f"Agent output didn't contain UUID or JSON: {content}")
                    
    except Exception as e:
        if final_id != "unknown_id" and proposed_tx is None and ("TaskGroup" in str(e) or "JSONRPCMessage" in str(e) or "BrokenResourceError" in str(e)):
            print("Ignored MCP shutdown error, agent succeeded.")
        else:
            return {"status": "error", "error": f"Actual Budget search failed: {str(e)}"}
            
    if proposed_tx:
        return {"proposed_transaction": proposed_tx}
    return {"actual_tx_id": final_id}

def human_approval_node(state: ReceiptState):
    """Placeholder node for human interrupt"""
    return {}

async def create_actual_transaction_node(state: ReceiptState):
    """Create the transaction in Actual Budget using an Agent"""
    final_id = "unknown_id"
    try:
        print("Creating transaction in Actual Budget via Agent...")
        proposed_tx = state.get("proposed_transaction")
        
        server_params = StdioServerParameters(
            command="node",
            args=["wrapper.js"],
            env=os.environ.copy()
        )
        
        llm = ChatOpenAI(
            model="deepseek-v4-flash", 
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""), 
            base_url="https://api.deepseek.com/v1"
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_tools = await load_mcp_tools(session)
                
                # Give it create tools!
                create_tools = [t for t in mcp_tools if "actual_transactions" in t.name or "actual_query" in t.name]
                agent = create_react_agent(llm, tools=create_tools)
                
                prompt = f"""
                You are a financial assistant connecting to Actual Budget.
                Please create the following transaction exactly as specified:
                {proposed_tx}
                
                Task:
                1. Create a new transaction using actual_transactions with the provided date, amount_cents, payee_name, account_id, and notes.
                2. After creation, you MUST re-query actual_query to get the final transaction ID of the transaction you just created.
                
                IMPORTANT: YOUR FINAL MESSAGE MUST BE EXACTLY THE 36-CHARACTER TRANSACTION UUID. Do not say "The ID is...". Just output the UUID string.
                """
                
                res = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
                content = res["messages"][-1].content.strip()
                
                import re
                match = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', content, re.IGNORECASE)
                if match:
                    final_id = match.group(0)
                    print(f"Agent created tx: {final_id}")
                else:
                    print(f"Agent output didn't contain UUID: {content}")
                    final_id = "unknown_id"
                    
    except Exception as e:
        if final_id != "unknown_id" and ("TaskGroup" in str(e) or "JSONRPCMessage" in str(e) or "BrokenResourceError" in str(e)):
            print("Ignored MCP shutdown error, agent succeeded.")
        else:
            return {"status": "error", "error": f"Actual Budget creation failed: {str(e)}"}
            
    return {"actual_tx_id": final_id}

async def tag_actual_transaction_node(state: ReceiptState):
    """Append #receipt to the matched/created transaction's notes in Actual Budget."""
    tx_id = state.get("actual_tx_id")
    if not tx_id or tx_id == "unknown_id":
        return {}

    try:
        print(f"Tagging transaction {tx_id} with #receipt...")
        server_params = StdioServerParameters(
            command="node",
            args=["wrapper.js"],
            env=os.environ.copy()
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                existing_notes = ""
                try:
                    res = await session.call_tool("actual_query", {"args": [
                        "run", "--table", "transactions", "--filter", json.dumps({"id": tx_id}), "--select", "id,notes", "--format", "json"
                    ]})
                    if res.content and len(res.content) > 0:
                        query_data = json.loads(res.content[0].text)
                        if query_data and len(query_data) > 0:
                            existing_notes = query_data[0].get("notes") or ""
                except Exception as e:
                    print(f"Failed to fetch existing notes: {e}")
                
                if "#receipt" not in existing_notes:
                    if existing_notes.strip():
                        new_notes = existing_notes + " #receipt"
                    else:
                        new_notes = "#receipt"
                else:
                    print(f"Transaction {tx_id} already has #receipt tag.")
                    return {}

                mcp_tools = await load_mcp_tools(session)
                tx_tool = next((t for t in mcp_tools if t.name == "actual_transactions"), None)
                if tx_tool:
                    await tx_tool.ainvoke({"args": [
                        "update", tx_id,
                        "--data", json.dumps({"notes": new_notes})
                    ]})
                    print(f"✅ Tagged transaction {tx_id}")
    except Exception as e:
        if "TaskGroup" in str(e) or "JSONRPCMessage" in str(e) or "BrokenResourceError" in str(e):
            print("Ignored MCP shutdown error, tag succeeded.")
        else:
            print(f"⚠️ Failed to tag transaction (non-fatal): {e}")
    return {}

def save_pocketbase_node(state: ReceiptState):
    """Save the receipt and items to PocketBase"""
    try:
        print("Saving to PocketBase...")
        parsed = state.get("parsed_data")
        tx_id = state.get("actual_tx_id")
        
        if not parsed or not tx_id or tx_id == "unknown_id":
            return {"status": "error", "error": "Missing parsed data or valid tx_id (was unknown_id)"}
            
        date = parsed["date"]
        amount = parsed["total"]
        payee = parsed["payee"]
        lines = parsed.get("items", [])
        source_ref = state.get("gdrive_path", "")
        image_paths = state.get("image_paths", [])
        file_path = state.get("file_path", "")
        source_hash = _file_hash(file_path) if file_path else ""
        
        res = save_receipt(tx_id, lines, date, source_ref, amount, payee, source_hash, image_paths)
        
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "error": f"PocketBase save failed: {str(e)}"}

def router_after_dedup(state: ReceiptState):
    if state.get("status") == "already_processed":
        return "skip"
    return "continue"

def router_check_error(state: ReceiptState):
    if state.get("status") == "error":
        return "error"
    return "continue"

def router_after_search(state: ReceiptState):
    if state.get("status") == "error":
        return "error"
    if state.get("proposed_transaction") or state.get("actual_tx_id") == "unknown_id":
        return "human_approval"
    return "save_pocketbase"

def router_after_human(state: ReceiptState):
    if state.get("actual_tx_id") and state.get("actual_tx_id") != "unknown_id":
        return "save_pocketbase"
    if state.get("human_approved"):
        return "create_actual"
    return "skip"

from langgraph.checkpoint.memory import MemorySaver

# Build Graph
workflow = StateGraph(ReceiptState)

workflow.add_node("dedup_check", skip_if_processed)
workflow.add_node("extract_ocr", extract_ocr_node)
workflow.add_node("parse_receipt", parse_receipt_node)
workflow.add_node("search_actual", search_actual_transaction_node)
workflow.add_node("human_approval", human_approval_node)
workflow.add_node("create_actual", create_actual_transaction_node)
workflow.add_node("tag_actual", tag_actual_transaction_node)
workflow.add_node("save_pocketbase", save_pocketbase_node)

workflow.set_entry_point("dedup_check")

workflow.add_conditional_edges("dedup_check", router_after_dedup, {"skip": END, "continue": "extract_ocr"})
workflow.add_conditional_edges("extract_ocr", router_check_error, {"error": END, "continue": "parse_receipt"})
workflow.add_conditional_edges("parse_receipt", router_check_error, {"error": END, "continue": "search_actual"})
workflow.add_conditional_edges("search_actual", router_after_search, {"error": END, "human_approval": "human_approval", "save_pocketbase": "tag_actual"})
workflow.add_conditional_edges("human_approval", router_after_human, {"create_actual": "create_actual", "save_pocketbase": "tag_actual", "skip": END})
workflow.add_conditional_edges("create_actual", router_check_error, {"error": END, "continue": "tag_actual"})
workflow.add_edge("tag_actual", "save_pocketbase")
workflow.add_edge("save_pocketbase", END)

memory = MemorySaver()
receipt_processor = workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])
