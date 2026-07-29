import os
import json
import base64
from typing import TypedDict, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from pdf2image import convert_from_path

from actual_orch.clients.cache import get_ocr, set_ocr, get_parsed, set_parsed, get_classify, set_classify
from actual_orch.nodes.pocketbase import skip_if_processed, save_pocketbase_node
from actual_orch.nodes.actual_budget import search_deterministic_node, search_agent_node, tag_actual_transaction_node, save_actual_transaction_node, estimate_tokens
import datetime

class ReceiptState(TypedDict, total=False):
    file_path: str        # Local path to the downloaded PDF or PNG
    source_ref: str       # Original source reference (e.g. gdrive path or email msg_id)
    source: str           # Source type (e.g. gdrive, email)
    ocr_text: str         # Text extracted by Gemini
    parsed_data: Dict[str, Any]
    actual_tx_id: str
    actual_tx_id_transfer: str
    status: str
    error: str
    email_date: str
    email_sender: str
    email_subject: str
    email_folder: str
    image_paths: List[str] # Paths to extracted JPEG images
    proposed_transaction: str # JSON string of proposed transaction if not found
    human_approved: bool
    not_a_receipt: bool   # True if content was classified as not a receipt
    forex_receipt: bool   # True if currency is not CAD
    parsing_failed: bool  # True if deepseek parsing failed entirely
    parse_retries: int    # Number of times parsing has been retried
    log_agent: bool       # True if agent logs should be printed
    create_missing: bool  # True if we should auto-create a transaction on no-match
    classification: str   # The result of the LLM classification (e.g. RECEIPT, INVOICE)

# Pydantic model for Deepseek parsing
class ReceiptItem(BaseModel):
    description: str = Field(description="Description of the item")
    amount: float = Field(description="Amount of the item")

class ParsedReceipt(BaseModel):
    date: str = Field(description="Date of the receipt in YYYY-MM-DD format")
    total_cad: Optional[float] = Field(default=None, description="Total amount of the receipt in CAD ONLY if explicitly printed on the receipt. Do NOT calculate currency conversions yourself. If the receipt is in a foreign currency, leave this as null.")
    currency: str = Field(description="Currency code (e.g. CAD, USD, EUR, JPY, GBP). Infer from symbols (e.g. € or ¥), text/language, or addresses. If clearly from Canada or CAD is not explicitly stated otherwise, use CAD.")
    total_foreign_currency: Optional[float] = Field(default=None, description="If the receipt is clearly in a foreign currency (not CAD), the exact total amount in that foreign currency. Otherwise leave as null.")
    payee: str = Field(description="Name of the store or payee")
    items: List[ReceiptItem] = Field(description="List of individual items on the receipt")
    account_key: Optional[str] = Field(default=None, description="If a list of valid account keys is provided, the most likely account key for this receipt based on the OCR text. Otherwise null.")
    category_id: Optional[str] = Field(default=None, description="If a list of categories is provided, the most likely category ID for this receipt. Otherwise null.")

def extract_ocr_node(state: ReceiptState):
    """Extract raw text from the PDF or image using Gemini Vision"""
    try:
        file_path = state["file_path"]

        # If it's a PDF, we need page images for PocketBase upload and Gemini
        image_paths = []
        if file_path.lower().endswith(".pdf"):
            gen_dir = os.path.join(os.getcwd(), "cache", "generated_image")
            os.makedirs(gen_dir, exist_ok=True)
            images = convert_from_path(file_path)
            for idx, img in enumerate(images):
                safe_name = os.path.basename(file_path) + f"_page_{idx}.jpg"
                img_path = os.path.join(gen_dir, safe_name)
                if not os.path.exists(img_path):
                    img.save(img_path, format="JPEG")
                image_paths.append(img_path)
        else:
            # It's already an image (jpg/jpeg/png)
            image_paths.append(file_path)
            
        # Check if ocr_text is already provided (e.g. for emails)
        prefix = f"receipt_{state.get('source', '')}_" if state.get('source') else "receipt_"
        if "ocr_text" in state and state["ocr_text"]:
            meta = {
                "response_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "model": "email_client_parser",
                "schema_version": "1.0.0"
            }
            set_ocr(file_path, state["ocr_text"], meta, prefix=prefix)
            return {"ocr_text": state["ocr_text"], "image_paths": image_paths}

        # Check Redis cache
        cached = get_ocr(file_path, prefix=prefix)
        if cached is None and prefix != "receipt_":
            cached = get_ocr(file_path, prefix="receipt_")
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
            
            ext = img_path.lower().split(".")[-1]
            mime_type = "image/png" if ext == "png" else "image/jpeg"
            
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{img_str}"}
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
            
        meta = {
            "response_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": "gemini-3.1-flash-lite",
            "schema_version": "1.0.0"
        }
        set_ocr(file_path, ocr_text, meta, prefix=prefix)
            
        return {"ocr_text": ocr_text, "image_paths": image_paths, "not_a_receipt": False}
    except Exception as e:
        return {"status": "error", "error": f"OCR failed: {str(e)}"}

def classify_input_node(state: ReceiptState):
    """Classify whether the content is actually a receipt."""
    try:
        if not state.get("ocr_text"):
            return {"status": "error", "error": "No OCR text found"}

        file_path = state["file_path"]
        prefix = f"receipt_{state.get('source', '')}_" if state.get('source') else "receipt_"
        
        classification = get_classify(file_path, prefix=prefix)
        if classification is None and prefix != "receipt_":
            classification = get_classify(file_path, prefix="receipt_")
            
        if classification is None:
            llm = ChatOpenAI(
                model="deepseek-v4-flash", 
                api_key=os.environ.get("DEEPSEEK_API_KEY", ""), 
                base_url="https://api.deepseek.com/v1",
                max_tokens=64
            )
            
            system_prompt = (
                "You are a document classifier. Given raw OCR text, classify the document into EXACTLY one of these categories:\n\n"
                "RECEIPT - A document confirming a completed payment, charge, or money transfer. "
                "Examples: store receipts, payment confirmations, Interac e-Transfer confirmations, ATM receipts, bank transfer records, money transfer screenshots.\n\n"
                "INVOICE - A bill or notice showing an amount owed that has NOT been charged or paid yet. "
                "Examples: utility bills, tax bills, property tax notices, insurance statements, subscription invoices with a due date.\n\n"
                "ORDER_CONFIRMATION - An order acknowledgment or shipping notification. "
                "Examples: online order confirmations, delivery notifications, booking confirmations.\n\n"
                "NOT_RECEIPT - Anything that is not a financial document. "
                "Examples: newsletters, advertisements, general correspondence, account statements with no actionable amount.\n\n"
                "Respond with EXACTLY one of: RECEIPT, INVOICE, ORDER_CONFIRMATION, or NOT_RECEIPT. Do not explain."
            )
            
            client_ctx = "\n\nCLIENT CONTEXT:\nThe client lives in Toronto, Ontario and mainly uses CAD."
            if state.get("source") == "email":
                email_ctx = f"\n\nEMAIL CONTEXT:\n- Date: {state.get('email_date', 'Unknown')}\n- Sender: {state.get('email_sender', 'Unknown')}\n- Subject: {state.get('email_subject', 'Unknown')}\n- Folder: {state.get('email_folder', 'Unknown')}"
                system_prompt += email_ctx + client_ctx
            else:
                system_prompt += client_ctx
                
            classify_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{ocr_text}")
            ])
            classify_chain = classify_prompt | llm
            
            prompt_val = classify_prompt.format_prompt(ocr_text=state["ocr_text"])
            print(f"  [Token Est] Classify prompt: ~{estimate_tokens(prompt_val.to_string())} tokens")
            
            classify_result = classify_chain.invoke({"ocr_text": state["ocr_text"]})
            classification = classify_result.content.strip().upper()
            
            meta = {
                "response_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "model": "deepseek-v4-flash",
                "schema_version": "1.0.0"
            }
            set_classify(file_path, classification, meta, prefix=prefix)
        else:
            print(f"Using cached classification data (Redis): {classification}")
        
        if "NOT_RECEIPT" in classification or "ORDER_CONFIRMATION" in classification or "INVOICE" in classification:
            print(f"  [Parse] Document classified as {classification}. Skipping completely.")
            return {"not_a_receipt": True, "classification": classification}
            
        is_future_date = False

        parsed_data = state.get("parsed_data")
        if not parsed_data:
            cached_parsed = get_parsed(file_path, prefix=prefix)
            if cached_parsed is None and prefix != "receipt_":
                cached_parsed = get_parsed(file_path, prefix="receipt_")
            parsed_data = cached_parsed
            
        if parsed_data and parsed_data.get("date"):
            try:
                parsed_date = datetime.datetime.strptime(parsed_data["date"], "%Y-%m-%d").date()
                if datetime.date.today() < parsed_date:
                    is_future_date = True
            except Exception:
                pass
                
        if is_future_date:
            print(f"  ⏭️  Skipped {state.get('source_ref')}: Receipt date is in the future. (Classified as: {classification})")
            return {"not_a_receipt": True, "classification": classification}
            
        return {"not_a_receipt": False, "classification": classification}
    except Exception as e:
        return {"status": "error", "error": f"Classification failed: {str(e)}"}

async def parse_receipt_node(state: ReceiptState):
    """Parse the OCR text using DeepSeek."""
    try:
        file_path = state["file_path"]
        prefix = f"receipt_{state.get('source', '')}_" if state.get('source') else "receipt_"
        
        parse_retries = state.get("parse_retries", 0)
        
        if parse_retries == 0:
            cached = get_parsed(file_path, prefix=prefix)
            if cached is None and prefix != "receipt_":
                cached = get_parsed(file_path, prefix="receipt_")
            if cached is not None:
                print("Using cached parsed data (Redis)...")
                return {"parsed_data": cached}

        model_name = "deepseek-v4-pro" if parse_retries > 0 else "deepseek-v4-flash"
        print(f"Parsing receipt with {model_name} (retry {parse_retries})...")
        
        llm = ChatOpenAI(
            model=model_name, 
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""), 
            base_url="https://api.deepseek.com/v1",
            max_tokens=1024
        )
        
        from actual_orch.shared.peak_hours import pause_if_peak_hours
        pause_if_peak_hours("deepseek")
        
        source_type = state.get("source")
        if source_type == "gdrive":
            source_desc = "OCR text from a paper receipt"
        elif source_type == "email":
            source_desc = "raw text from an email receipt"
        else:
            source_desc = "OCR text from a receipt"
            
        system_prompt = "You are an expert at extracting structured data from {source_desc}. Extract the date, total amount, currency, payee, and line items. If a date is missing, use the current date or guess based on context. Amount should be a number. Date MUST be YYYY-MM-DD. You MUST respond with a JSON object. For the top-level keys, strictly use 'date', 'total_cad', 'currency', 'total_foreign_currency', 'payee', 'items', 'account_key', and 'category_id'. For the items array, strictly use the keys 'description' and 'amount'. For ATM receipts or transfers, 'items' can be empty, and 'payee' can be the bank name or 'ATM'. For the item 'description', write a casual, human-readable description but try to include all useful info (like brand or type) while avoiding weird characters or abbreviations. IMPORTANT: Include taxes (HST, GST, PST, VAT, etc.) and points/rewards redemptions as separate line items. Points redemptions should appear as negative amounts (e.g. -2.50 for a $2.50 points discount). The line items should add up to the receipt total. IMPORTANT: Your default assumption for currency MUST be CAD. Unless there is definitive proof of a foreign transaction (e.g., explicitly stating USD or foreign currency symbol), you must output CAD. Payees like Uber (Amsterdam) or Apple (California) often show foreign addresses but process transactions in CAD. Do NOT assume USD just because of a foreign address. If it is CAD, leave 'total_foreign_currency' null."

        client_ctx = "\n\nCLIENT CONTEXT:\nThe client lives in Toronto, Ontario and mainly uses CAD."
        if source_type == "email":
            email_ctx = f"\n\nEMAIL CONTEXT:\n- Date: {state.get('email_date', 'Unknown')}\n- Sender: {state.get('email_sender', 'Unknown')}\n- Subject: {state.get('email_subject', 'Unknown')}\n- Folder: {state.get('email_folder', 'Unknown')}"
            system_prompt += email_ctx + client_ctx
        else:
            system_prompt += client_ctx

        if state.get("create_missing"):
            try:
                from actual_orch.clients.mcp import ActualMCPClient
                mapping_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../actual-utils/cache/account-mapping.json"))
                mapping = {}
                if os.path.exists(mapping_path):
                    with open(mapping_path, "r") as f:
                        mapping = json.load(f)
                
                session = await ActualMCPClient.get_session()
                res = await session.call_tool("actual_accounts", {"args": ["list", "--format", "json"]})
                if res.content:
                    try:
                        accounts = json.loads(res.content[0].text)
                        on_budget_ids = {acc.get("id") for acc in accounts if not acc.get("offbudget") and isinstance(acc, dict) and acc.get("id")}
                        valid_keys = [k for k, v in mapping.items() if v in on_budget_ids]
                        
                        if valid_keys:
                            keys_json = json.dumps(valid_keys).replace("{", "{{").replace("}", "}}")
                            system_prompt += f"\n\nACCOUNT SELECTION:\nYou must also pick the MOST LIKELY account key from this list of valid on-budget account keys: {keys_json}\nValid account keys generally fall into two categories:\n1. The last 4 digits of a credit card (e.g. 1234, 5234).\n2. Different account names or types (e.g. WS-CHEQUING, DEFAULT-CC).\nIf you see the last 4 digits of a card on the receipt that matches one of the keys, use that key for 'account_key'. If you see the name of an institution or account type that strongly matches, use that key. If you are unsure, output the key 'DEFAULT-CC' if it exists in the list, or just pick the most likely one."
                    except Exception as json_e:
                        print(f"  ⚠️ Failed to parse accounts JSON: {json_e}")
                
                categories_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../actual-utils/cache/categories.json"))
                if os.path.exists(categories_path):
                    with open(categories_path, "r") as f:
                        categories = json.load(f)
                    if isinstance(categories, list):
                        valid_categories = [{"id": c.get("id"), "name": c.get("name")} for c in categories if isinstance(c, dict) and not c.get("hidden") and not c.get("is_income") and c.get("id")]
                        if valid_categories:
                            categories_json = json.dumps(valid_categories).replace("{", "{{").replace("}", "}}")
                            system_prompt += f"\n\nCATEGORY SELECTION:\nYou must also pick the MOST LIKELY category ID from this list: {categories_json}\nBased on the receipt items and payee, choose the best category and output its 'id' for the 'category_id' field. If unsure, pick the most generic expense category."
            except Exception as e:
                print(f"  ⚠️ Failed to load on-budget accounts for prompt: {e}")

        if parse_retries > 0:
            prev_date = state.get("parsed_data", {}).get("date", "unknown")
            system_prompt += f"\n\nCRITICAL: Double check the transaction date is correct! You previously extracted {prev_date} but we couldn't find a matching transaction in the bank records around that date. Ensure you aren't using the print date or statement date instead of the actual purchase date. Look closely for the date of the actual transaction."

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Here is the {source_desc}:\n\n{ocr_text}")
        ])
        
        prompt_val = prompt.format_prompt(source_desc=source_desc, ocr_text=state["ocr_text"])
        print(f"  [Token Est] Extract prompt: ~{estimate_tokens(prompt_val.to_string())} tokens")

        try:
            structured_llm = llm.with_structured_output(ParsedReceipt, method="json_mode")
            chain = prompt | structured_llm
            result = await chain.ainvoke({"ocr_text": state["ocr_text"], "source_desc": source_desc})
            parsed_dict = result.dict()
        except Exception as e:
            print(f"  ❌ Parsing failed: {e}. Flagging for review.")
            return {"parsing_failed": True}
        
        total_val = parsed_dict.pop("total_cad", None)
        parsed_dict["total"] = total_val if total_val is not None else 0.0
        
        if state.get("source") == "email" and state.get("email_date"):
            if parse_retries == 0:
                try:
                    email_dt = datetime.datetime.strptime(state["email_date"], "%Y-%m-%d")
                    if parsed_dict.get("date"):
                        parsed_dt = datetime.datetime.strptime(parsed_dict["date"], "%Y-%m-%d")
                        if (email_dt - parsed_dt).days > 3:
                            print(f"  [Date] Overriding parsed date {parsed_dict['date']} with email date {state['email_date']} (>3 days older)")
                            parsed_dict["date"] = state["email_date"]
                    else:
                        parsed_dict["date"] = state["email_date"]
                except Exception as e:
                    print(f"  ⚠️ Could not compare dates for override: {e}")
        
        if "total_foreign_currency" in parsed_dict:
            parsed_dict["foreign_amount"] = parsed_dict.pop("total_foreign_currency")
        
        currency = parsed_dict.get("currency", "CAD")
        if currency:
            currency = currency.upper()
        else:
            currency = "CAD"
        forex_receipt = currency != "CAD"
        
        if forex_receipt and parsed_dict.get("foreign_amount"):
            try:
                import sys
                utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../actual-utils"))
                if utils_path not in sys.path:
                    sys.path.append(utils_path)
                from modules.market import get_price
                
                price, _ = get_price(currency, parsed_dict.get("date"))
                if price:
                    cad_equiv = parsed_dict["foreign_amount"] * price
                    parsed_dict["total"] = round(cad_equiv, 2)
                    print(f"  [Forex] Converted {parsed_dict['foreign_amount']} {currency} to approx {parsed_dict['total']} CAD")
            except Exception as e:
                print(f"  ⚠️ Could not convert foreign amount to CAD: {e}")
                
        meta = {
            "response_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": model_name,
            "schema_version": "1.0.0"
        }
        set_parsed(file_path, parsed_dict, meta, prefix=prefix)
        
        if parsed_dict.get("date"):
            try:
                parsed_date = datetime.datetime.strptime(parsed_dict["date"], "%Y-%m-%d").date()
                if datetime.date.today() < parsed_date:
                    print(f"  ❌ Parsed date {parsed_date} is in the future. Failing with error state.")
                    return {"status": "error", "error": f"Parsed date {parsed_date} is in the future (saved to cache)"}
            except Exception:
                pass
            
        return {"parsed_data": parsed_dict, "forex_receipt": forex_receipt, "parsing_failed": False}
    except Exception as e:
        return {"status": "error", "error": f"Parsing failed: {str(e)}"}


def human_approval_node(state: ReceiptState):
    """Placeholder node for human interrupt"""
    return {}

def router_after_dedup(state: ReceiptState):
    if state.get("status") == "already_processed":
        return "skip"
    return "continue"

def router_check_error(state: ReceiptState):
    if state.get("status") == "error":
        return "error"
    return "continue"

def router_after_classify(state: ReceiptState):
    if state.get("status") == "error":
        return "error"
    if state.get("not_a_receipt"):
        return "skip"
    return "continue"

def router_after_parse(state: ReceiptState):
    if state.get("status") == "error":
        return "error"
    if state.get("parsing_failed"):
        return "human_approval"
    return "continue"

def router_after_search_deterministic(state: ReceiptState):
    if state.get("status") == "error":
        return "error"
    if state.get("status") == "fallback_to_agent":
        return "fallback_to_agent"
    if state.get("actual_tx_id") != "unknown_id":
        return "save_pocketbase"
    return "error"

def router_after_search_agent(state: ReceiptState):
    if state.get("status") == "error":
        return "error"
    if state.get("status") == "retry_parse":
        return "retry_parse"
    if state.get("status") == "human_approval" or state.get("actual_tx_id") == "unknown_id":
        if state.get("create_missing"):
            return "save_actual"
        return "human_approval"
    return "save_pocketbase"

def router_after_human(state: ReceiptState):
    if state.get("actual_tx_id") and state.get("actual_tx_id") != "unknown_id":
        return "save_pocketbase"
    return "skip"

# Build Graph
workflow = StateGraph(ReceiptState)

workflow.add_node("skip_if_processed", skip_if_processed)
workflow.add_node("extract_ocr", extract_ocr_node)
workflow.add_node("classify_input", classify_input_node)
workflow.add_node("parse_receipt", parse_receipt_node)
workflow.add_node("search_deterministic", search_deterministic_node)
workflow.add_node("search_agent", search_agent_node)
workflow.add_node("human_approval", human_approval_node)
workflow.add_node("save_actual", save_actual_transaction_node)
workflow.add_node("tag_actual", tag_actual_transaction_node)
workflow.add_node("save_pocketbase", save_pocketbase_node)

workflow.set_entry_point("skip_if_processed")

workflow.add_conditional_edges("skip_if_processed", router_after_dedup, {"skip": END, "continue": "extract_ocr"})
workflow.add_conditional_edges("extract_ocr", router_check_error, {"error": END, "continue": "classify_input"})
workflow.add_conditional_edges("classify_input", router_after_classify, {"error": END, "human_approval": "human_approval", "continue": "parse_receipt", "skip": END})
workflow.add_conditional_edges("parse_receipt", router_after_parse, {"error": END, "human_approval": "human_approval", "continue": "search_deterministic"})
workflow.add_conditional_edges("search_deterministic", router_after_search_deterministic, {
    "error": END, 
    "fallback_to_agent": "search_agent", 
    "save_pocketbase": "save_pocketbase"
})
workflow.add_conditional_edges("search_agent", router_after_search_agent, {
    "error": END, 
    "human_approval": "human_approval", 
    "save_actual": "save_actual",
    "save_pocketbase": "save_pocketbase",
    "retry_parse": "parse_receipt"
})
workflow.add_conditional_edges("human_approval", router_after_human, {"save_pocketbase": "save_pocketbase", "skip": END})
workflow.add_edge("save_actual", "save_pocketbase")
workflow.add_edge("save_pocketbase", "tag_actual")
workflow.add_edge("tag_actual", END)
