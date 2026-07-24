import os
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
from actual_orch.nodes.actual_budget import search_actual_transaction_node, tag_actual_transaction_node, estimate_tokens
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
    image_paths: List[str] # Paths to extracted JPEG images
    proposed_transaction: str # JSON string of proposed transaction if not found
    human_approved: bool
    not_a_receipt: bool   # True if content was classified as not a receipt
    forex_receipt: bool   # True if currency is not CAD
    parsing_failed: bool  # True if deepseek parsing failed entirely

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

def parse_receipt_node(state: ReceiptState):
    """Parse the OCR text using DeepSeek. First classifies whether the content
    is actually a receipt; if not, flags it for human review."""
    try:
        if not state.get("ocr_text"):
            return {"status": "error", "error": "No OCR text found"}

        file_path = state["file_path"]

        # Check Redis cache
        prefix = f"receipt_{state.get('source', '')}_" if state.get('source') else "receipt_"
        cached = get_parsed(file_path, prefix=prefix)
        if cached is None and prefix != "receipt_":
            cached = get_parsed(file_path, prefix="receipt_")
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
        
        from actual_orch.shared.peak_hours import pause_if_peak_hours
        pause_if_peak_hours("deepseek")
        
        # ── Step 1: Classify whether this is actually a receipt ──
        classification = get_classify(file_path, prefix=prefix)
        if classification is None and prefix != "receipt_":
            classification = get_classify(file_path, prefix="receipt_")
            
        if classification is None:
            classify_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a document classifier. Given raw OCR text, determine if the content is a purchase receipt, invoice, ATM receipt, or bank transfer record (anything representing a financial transaction with a date and amount). Respond with EXACTLY one word: RECEIPT or NOT_RECEIPT. Do not explain."),
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
            print("Using cached classification data (Redis)...")
        
        if "NOT_RECEIPT" in classification:
            if state.get("source") == "email":
                print(f"  [Parse] Email classified as NOT a receipt. Skipping.")
            else:
                print(f"⚠️  Content classified as NOT a receipt. Flagging for human review.")
            return {"not_a_receipt": True}
        
        # ── Step 2: Structured extraction (only if it IS a receipt) ──
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at extracting structured data from raw OCR text of receipts. Extract the date, total amount, currency, payee, and line items. If a date is missing, use the current date or guess based on context. Amount should be a number. Date MUST be YYYY-MM-DD. You MUST respond with a JSON object. For the top-level keys, strictly use 'date', 'total_cad', 'currency', 'total_foreign_currency', 'payee', and 'items'. For the items array, strictly use the keys 'description' and 'amount'. For ATM receipts or transfers, 'items' can be empty, and 'payee' can be the bank name or 'ATM'. For the item 'description', write a casual, human-readable description but try to include all useful info (like brand or type) while avoiding weird characters or abbreviations. IMPORTANT: Be very careful about classifying foreign receipts. If you detect a non-Canadian address, foreign language, or foreign currency symbol, set the 'currency' appropriately (e.g., USD, EUR, JPY) and put the exact amount in 'total_foreign_currency'. If it is CAD, leave 'total_foreign_currency' null."),
            ("human", "Here is the raw OCR text:\n\n{ocr_text}")
        ])
        
        prompt_val = prompt.format_prompt(ocr_text=state["ocr_text"])
        print(f"  [Token Est] Extract prompt: ~{estimate_tokens(prompt_val.to_string())} tokens")

        try:
            structured_llm = llm.with_structured_output(ParsedReceipt, method="json_mode")
            chain = prompt | structured_llm
            result = chain.invoke({"ocr_text": state["ocr_text"]})
            parsed_dict = result.dict()
        except Exception as e:
            print(f"  ⚠️ Initial parsing failed: {e}. Retrying with deepseek-chat (pro model)...")
            try:
                pro_llm = ChatOpenAI(
                    model="deepseek-chat", 
                    api_key=os.environ.get("DEEPSEEK_API_KEY", ""), 
                    base_url="https://api.deepseek.com/v1",
                    max_tokens=1024
                )
                pro_structured_llm = pro_llm.with_structured_output(ParsedReceipt, method="json_mode")
                pro_chain = prompt | pro_structured_llm
                result = pro_chain.invoke({"ocr_text": state["ocr_text"]})
                parsed_dict = result.dict()
            except Exception as pro_e:
                print(f"  ❌ Pro model also failed: {pro_e}. Flagging for review.")
                return {"parsing_failed": True}
        
        total_val = parsed_dict.pop("total_cad", None)
        parsed_dict["total"] = total_val if total_val is not None else 0.0
        
        if state.get("source") == "email" and state.get("email_date"):
            parsed_dict["date"] = state["email_date"]
        
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
            "model": "deepseek-v4-flash" if "parsing_failed" not in locals() else "deepseek-chat",
            "schema_version": "1.0.0"
        }
        set_parsed(file_path, parsed_dict, meta, prefix=prefix)
            
        return {"parsed_data": parsed_dict, "forex_receipt": forex_receipt, "parsing_failed": False, "not_a_receipt": False}
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

def router_after_parse(state: ReceiptState):
    """Route after parse_receipt: non-receipts go to human review."""
    if state.get("status") == "error":
        return "error"
    if state.get("not_a_receipt") or state.get("parsing_failed"):
        if state.get("source") == "email":
            return "skip"
        return "human_approval"
    return "continue"

def router_after_search(state: ReceiptState):
    if state.get("status") == "error":
        return "error"
    if state.get("actual_tx_id") == "unknown_id":
        return "human_approval"
    return "save_pocketbase"

def router_after_human(state: ReceiptState):
    if state.get("actual_tx_id") and state.get("actual_tx_id") != "unknown_id":
        return "save_pocketbase"
    return "skip"

# Build Graph
workflow = StateGraph(ReceiptState)

workflow.add_node("dedup_check", skip_if_processed)
workflow.add_node("extract_ocr", extract_ocr_node)
workflow.add_node("parse_receipt", parse_receipt_node)
workflow.add_node("search_actual", search_actual_transaction_node)
workflow.add_node("human_approval", human_approval_node)
workflow.add_node("tag_actual", tag_actual_transaction_node)
workflow.add_node("save_pocketbase", save_pocketbase_node)

workflow.set_entry_point("dedup_check")

workflow.add_conditional_edges("dedup_check", router_after_dedup, {"skip": END, "continue": "extract_ocr"})
workflow.add_conditional_edges("extract_ocr", router_check_error, {"error": END, "continue": "parse_receipt"})
workflow.add_conditional_edges("parse_receipt", router_after_parse, {"error": END, "human_approval": "human_approval", "continue": "search_actual", "skip": END})
workflow.add_conditional_edges("search_actual", router_after_search, {"error": END, "human_approval": "human_approval", "save_pocketbase": "tag_actual"})
workflow.add_conditional_edges("human_approval", router_after_human, {"save_pocketbase": "tag_actual", "skip": END})
workflow.add_edge("tag_actual", "save_pocketbase")
workflow.add_edge("save_pocketbase", END)
