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
from actual_orch.nodes.actual_budget import search_deterministic_node, search_agent_node, update_actual_transaction_node, save_actual_transaction_node, estimate_tokens
import datetime

class WorkflowStatus:
    """
    Class representing the flow control state for the LangGraph workflow.
    """
    CONTINUE = "continue"                     # Default state, proceed to the next logical node
    ERROR = "error"                           # An error occurred, routing to END or an error handler
    ALREADY_PROCESSED = "already_processed"   # The document was already processed, skip further steps
    FALLBACK_TO_AGENT = "fallback_to_agent"   # Deterministic search failed, falling back to agent search
    RETRY_PARSE = "retry_parse"               # Agent could not find transaction, retry parsing with better model
    HUMAN_APPROVAL = "human_approval"         # Needs human intervention/approval

class ProposedTransaction(TypedDict, total=False):
    date: str
    account: str
    payee: str
    notes: str
    category: str
    amount: float
    transfer_account: str

class AgentAttempt(TypedDict, total=False):
    node: str
    model: str
    success: bool
    error: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    response_time_ms: Optional[int]

class ReceiptState(TypedDict, total=False):
    file_path: str        # Local path to the downloaded PDF or PNG
    source_ref: str       # Original source reference (e.g. rclone/local path or email msg_id)
    source: str           # Source type (e.g. file, email, icloud, gdrive)
    ocr_text: str         # Text extracted by Gemini
    parsed_data: Dict[str, Any]
    actual_tx_id: str
    replaced_tx_id: str   # If the tx_id changed (e.g. recreating as split), this holds the old ID
    actual_tx_id_transfer: str
    update_transaction: ProposedTransaction # Agent proposed updates
    status: str
    error: str
    email_date: str
    email_context: str    # Context string containing sender, subject, folder, etc.
    image_paths: List[str] # Paths to extracted JPEG images
    proposed_transactions: List[ProposedTransaction] 
    agent_attempts: List[AgentAttempt] # History of all agent attempts (node, model, success, error)
    flag_log_agent: bool       # True if agent logs should be printed
    flag_create_missing: bool  # True if we should auto-create a transaction on no-match
    classification: str   # The result of the LLM classification (e.g. RECEIPT, INVOICE)
    override_classification: str # Explicit override for classification
    flag_skip_cache_types: List[str] # Cache types to bypass (e.g. 'ocr', 'classify', 'parsed')
    
# Pydantic model for Deepseek parsing
class ReceiptItem(BaseModel):
    description: str = Field(description="Description of the item")
    amount: float = Field(description="Amount of the item")

class ParsedProposedTransaction(BaseModel):
    amount: float = Field(description="Amount for this transaction line. Positive = expense, negative = deposit, refund, credit, or points redemption.")
    notes: str = Field(description="Concise description of this line (e.g. 'Groceries', 'Points redemption'). Do NOT list out items, summarize as short as possible.")
    category: Optional[str] = Field(default=None, description="The most likely category name from the provided list, or null if it's a transfer")
    transfer_account: Optional[str] = Field(default=None, description="The name of the account to transfer to, if this transaction is a transfer to another account")

class ParsedReceipt(BaseModel):
    date: str = Field(description="Date of the receipt in YYYY-MM-DD format")
    currency: str = Field(description="Currency code (e.g. CAD, USD, EUR, JPY, GBP). Infer from symbols (e.g. € or ¥), text/language, or addresses. Default to CAD unless there is definitive proof otherwise.")
    payee: str = Field(description="Name of the store or payee")
    notes: str = Field(default="", description="A concise ~50 character human-readable summary of what was purchased (e.g. 'Groceries - milk, eggs, bread' or 'Uber ride downtown'). Do NOT include the payee name, date, or amount — just describe the purchase.")
    items: List[ReceiptItem] = Field(description="List of individual items on the receipt")
    account_key: Optional[str] = Field(default=None, description="If a list of valid account keys is provided, the most likely account key for this receipt based on the OCR text. Otherwise null.")
    proposed_transactions: List[ParsedProposedTransaction] = Field(description="List of proposed transactions to create. The sum of amounts across all proposed transactions equals the receipt total.")

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
        cached = None
        if "ocr" not in state.get("flag_skip_cache_types", []):
            cached = get_ocr(file_path, prefix=prefix)
            if cached is None and prefix != "receipt_":
                cached = get_ocr(file_path, prefix="receipt_")
        
        if cached is not None:
            print("[extract_ocr][cache] " + "Using cached OCR text (Redis)...")
            return {"ocr_text": cached, "image_paths": image_paths}

        print(f"[extract_ocr] Extracting OCR with Gemini Vision for {file_path}...")
        
        # Respect Gemini's 15 Requests Per Minute (RPM) free tier rate limit
        import time
        time.sleep(4.1) 
        
        from actual_orch.shared.utils import get_agent_config
        attempts = get_agent_config("receipt_extract").get("attempts", [])
        model_name = attempts[0].get("model", "gemini-3.1-flash-lite") if attempts else "gemini-3.1-flash-lite"

        llm = ChatGoogleGenerativeAI(
            model=model_name,
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
        
        if state.get("flag_log_agent"):
            print(f"\n[extract_ocr][agent] --- AGENT PROMPT (gemini-3.1-flash-lite) ---")
            print("[extract_ocr] " + "Extract all the text from this receipt accurately, line by line. Output ONLY the raw text without any markdown formatting or extra commentary.")
            print(f"[extract_ocr] [{len(image_paths)} images attached]")
            print("[extract_ocr] " + "---------------------------------------\n")
        
        result = llm.invoke(messages)
        ocr_text = result.content
        if isinstance(ocr_text, list):
            ocr_text = "".join([block.get("text", "") if isinstance(block, dict) else str(block) for block in ocr_text])
        elif not isinstance(ocr_text, str):
            ocr_text = str(ocr_text)
            
        if state.get("flag_log_agent"):
            print(f"\n[extract_ocr][agent] --- AGENT RESPONSE (gemini-3.1-flash-lite) ---")
            print(f"[extract_ocr] {ocr_text}")
            print("[extract_ocr] " + "---------------------------------------\n")
            
        meta = {
            "response_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": "gemini-3.1-flash-lite",
            "schema_version": "1.0.0"
        }
        set_ocr(file_path, ocr_text, meta, prefix=prefix)
            
        return {"ocr_text": ocr_text, "image_paths": image_paths}
    except Exception as e:
        return {"status": WorkflowStatus.ERROR, "error": f"OCR failed: {str(e)}"}

def classify_input_node(state: ReceiptState):
    """Classify whether the content is actually a receipt."""
    try:
        if not state.get("ocr_text"):
            return {"status": WorkflowStatus.ERROR, "error": "No OCR text found"}

        file_path = state["file_path"]
        prefix = f"receipt_{state.get('source', '')}_" if state.get('source') else "receipt_"
        
        classification = state.get("override_classification")
        
        if classification:
            print(f"[classify_input] Using override classification: [{classification}]")
        else:
            if "classify" not in state.get("flag_skip_cache_types", []):
                classification = get_classify(file_path, prefix=prefix)
                if (not classification or not classification.strip()) and prefix != "receipt_":
                    classification = get_classify(file_path, prefix="receipt_")
                
        if not classification or not classification.strip():
            from actual_orch.shared.utils import get_agent_config
            attempts = get_agent_config("receipt_classify").get("attempts", [])
            model_name = attempts[0].get("model", "deepseek-v4-flash") if attempts else "deepseek-v4-flash"

            llm = ChatOpenAI(
                model=model_name, 
                api_key=os.environ.get("DEEPSEEK_API_KEY", ""), 
                base_url="https://api.deepseek.com/v1",
                max_tokens=64
            )
            
            config_path_class = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/classification.json"))
            classifications = {}
            if os.path.exists(config_path_class):
                with open(config_path_class, "r") as f:
                    class_data = json.load(f)
                    classifications = {k: v.get("prompt_description", "") for k, v in class_data.items() if isinstance(v, dict)}
            else:
                print(f"[classify_input][error] Missing {config_path_class}")
            
            system_prompt = "You are a document classifier. Given raw OCR text, classify the document into EXACTLY one of these categories:\n\n"
            for k, v in classifications.items():
                system_prompt += f"{k} - {v}\n\n"
            
            class_list = list(classifications.keys())
            class_str = ", ".join(class_list[:-1]) + f", or {class_list[-1]}" if len(class_list) > 1 else (class_list[0] if class_list else "")
            system_prompt += f"Respond with EXACTLY one of: {class_str}. Do not explain."
            
            client_ctx = "\n\nCLIENT CONTEXT:\nThe client lives in Toronto, Ontario and mainly uses CAD."
            try:
                config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../actual-utils/config.json"))
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config_data = json.load(f)
                    if config_data.get("AGENT_USER_CONTEXT"):
                        client_ctx += f"\n{config_data['AGENT_USER_CONTEXT']}"
            except Exception as e:
                print(f"[classify_input][error]   ⚠️ Failed to load AGENT_USER_CONTEXT: {e}")
            from actual_orch.shared.utils import get_receipt_type
            if get_receipt_type(state.get("source")) == "email":
                email_ctx = state.get("email_context", "")
                if state.get('email_date'):
                    email_ctx = f"- Date: {state.get('email_date')}\n" + email_ctx
                system_prompt += f"\n\nEMAIL CONTEXT:\n{email_ctx}" + client_ctx
            else:
                system_prompt += client_ctx
                
            classify_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{ocr_text}")
            ])
            classify_chain = classify_prompt | llm
            
            prompt_val = classify_prompt.format_prompt(ocr_text=state["ocr_text"])
            print(f"[classify_input][agent]  Classify prompt: ~{estimate_tokens(prompt_val.to_string())} tokens")
            
            if state.get("flag_log_agent"):
                print(f"\n[classify_input][agent] --- AGENT PROMPT (deepseek-v4-flash) ---")
                print(f"[classify_input] {prompt_val.to_string()}")
                print("[classify_input] " + "---------------------------------------\n")
            
            classify_result = classify_chain.invoke({"ocr_text": state["ocr_text"]})
            classification = classify_result.content.strip().upper()
            
            if state.get("flag_log_agent"):
                print(f"\n[classify_input][agent] --- AGENT RESPONSE (deepseek-v4-flash) ---")
                print(f"[classify_input] {classification}")
                print("[classify_input] " + "---------------------------------------\n")
            
            meta = {
                "response_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "model": "deepseek-v4-flash",
                "schema_version": "1.0.0"
            }
            set_classify(file_path, classification, meta, prefix=prefix)
        else:
            if not state.get("override_classification"):
                print(f"[classify_input][cache] Using cached classification data (Redis): [{classification.strip()}]")
        
        config_path_class = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/classification.json"))
        should_parse = True
        skip_if_future_date = False
        
        if os.path.exists(config_path_class):
            with open(config_path_class, "r") as f:
                class_data = json.load(f)
                for cls_name, cls_config in class_data.items():
                    if cls_name == classification.strip():
                        should_parse = cls_config.get("parse", True)
                        skip_if_future_date = cls_config.get("skip_if_future_date", False)
                        break
                
        if not should_parse:
            print(f"[classify_input][skip]   Document classified as {classification}. Skipping completely.")
            return {"classification": classification, "status": WorkflowStatus.ALREADY_PROCESSED}
            
        is_future_date = False

        parsed_data = state.get("parsed_data")
        if not parsed_data:
            cached_parsed = None
            if "parsed" not in state.get("flag_skip_cache_types", []):
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
                
        if is_future_date and skip_if_future_date:
            print(f"[classify_input][skip]   ⏭️  Skipped {state.get('source_ref')}: Receipt date is in the future. (Classified as: {classification})")
            return {"classification": classification, "status": WorkflowStatus.ALREADY_PROCESSED}
            
        print(f"[classify_input]   Document classified as: {classification}")
        return {"classification": classification}
    except Exception as e:
        return {"status": WorkflowStatus.ERROR, "error": f"Classification failed: {str(e)}"}

async def parse_receipt_node(state: ReceiptState):
    """Parse the OCR text using DeepSeek."""
    try:
        file_path = state["file_path"]
        prefix = f"receipt_{state.get('source', '')}_" if state.get('source') else "receipt_"
        
        agent_attempts = state.get("agent_attempts", [])
        parse_attempts = [a for a in agent_attempts if a.get("node") == "receipt_parse"]
        parse_retries = len(parse_attempts)
        
        from actual_orch.shared.utils import get_agent_config
        parsing_config = get_agent_config("receipt_parse")
        attempts_config = parsing_config.get("attempts", [{"model": "deepseek-v4-pro"}])
        max_attempts = len(attempts_config) if attempts_config else 1
        all_models_exhausted = parse_retries >= max_attempts
        
        # Try cache first (only on first attempt or if all models exhausted)
        parsed_dict = None
        model_name = None
        if parse_retries == 0 or all_models_exhausted:
            cached = None
            if "parsed" not in state.get("flag_skip_cache_types", []):
                cached = get_parsed(file_path, prefix=prefix)
                if cached is None and prefix != "receipt_":
                    cached = get_parsed(file_path, prefix="receipt_")
                    
            if cached is not None:
                if all_models_exhausted:
                    print("[parse_receipt][cache] " + "All models exhausted, using cached parsed data (Redis) as fallback...")
                else:
                    print("[parse_receipt][cache] " + "Using cached parsed data (Redis)...")
                parsed_dict = cached

        if parsed_dict is None:
            current_attempt = attempts_config[min(parse_retries, max_attempts - 1)] if attempts_config else {"model": "deepseek-v4-pro"}
            model_name = current_attempt.get("model", "deepseek-v4-pro")
            print(f"[parse_receipt] Parsing receipt with {model_name} (retry {parse_retries})...")
            
            llm = ChatOpenAI(
                model=model_name, 
                api_key=os.environ.get("DEEPSEEK_API_KEY", ""), 
                base_url="https://api.deepseek.com/v1",
                max_tokens=1024,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            
            from actual_orch.shared.peak_hours import pause_if_peak_hours
            pause_if_peak_hours("deepseek")
            
            from actual_orch.shared.utils import get_receipt_type
            source_type = get_receipt_type(state.get("source"))
            if source_type == "email":
                source_desc = "raw text from an email receipt"
            else:
                source_desc = "OCR text from a paper receipt or file"
                
            system_prompt = (
                "You are an expert at extracting structured data from {source_desc}. "
                "Respond with a JSON object using these top-level keys: 'date', 'currency', 'payee', 'notes', 'items', 'account_key', 'proposed_transactions'.\n\n"
                "RULES:\n"
                "- 'date': YYYY-MM-DD format. Today's date is {todays_date}. If missing, use the current date or guess from context. The date MUST NOT be in the future (after {todays_date}). If the parsed date would be in the future and email context is available, use the email date instead.\n"
                "- 'currency': The 3-letter currency code (e.g. CAD, USD, EUR, JPY). Default to CAD unless there is definitive proof otherwise (explicit USD symbol, foreign currency text). Payees with foreign addresses (Uber Amsterdam, Apple California) often charge in CAD — do NOT assume USD from an address alone.\n"
                "- 'payee': Name of the store or payee. For ATM receipts use the bank name or 'ATM'.\n"
                "- 'notes': A concise ~50 character summary of the purchase (e.g. 'Groceries - milk, eggs, bread'). Do NOT include payee, date, or amount. For ATM use 'ATM withdrawal'. For e-transfers, include the reference number if available (e.g., 'e-Transfer Ref: CA4ABCD123').\n"
                "- 'items': Array of line items with 'description' and 'amount'. Include taxes (HST/GST/PST/VAT) as separate items. Points/rewards redemptions should be negative amounts. Items can be empty for ATM/transfers.\n"
                "  - 'description': casual, human-readable — include useful info (brand, type) but avoid weird characters or abbreviations.\n\n"
                "PROPOSED TRANSACTIONS (most important):\n"
                "'proposed_transactions' is the primary output. Each entry has: 'amount' (number), 'notes' (concise description: do NOT list out items, summarize as short as possible), and EITHER 'category' (from the category list) OR 'transfer_account' (from the account list) if it is a transfer.\n"
                "- Most receipts have ONE proposed transaction with the receipt total as the amount.\n"
                "- If the receipt is for a deposit, refund, or credit (money coming INTO the account), the proposed_transaction 'amount' MUST be negative.\n"
                "- If the receipt includes a points redemption or refund (negative amount on the receipt), create a SEPARATE proposed transaction for it with a negative amount and an Income category.\n"
                "- The sum of all proposed_transaction amounts must equal the receipt total.\n"
                "- Amounts are in the receipt's currency (the 'currency' field).\n"
            )

            client_ctx = "\n\nCLIENT CONTEXT:\nThe client lives in Toronto, Ontario and mainly uses CAD."
            try:
                config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../actual-utils/config.json"))
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config_data = json.load(f)
                    if config_data.get("AGENT_USER_CONTEXT"):
                        client_ctx += f"\n{config_data['AGENT_USER_CONTEXT']}"
            except Exception as e:
                print(f"[parse_receipt][error]   ⚠️ Failed to load AGENT_USER_CONTEXT: {e}")
            if source_type == "email":
                email_ctx = state.get("email_context", "")
                if state.get('email_date'):
                    email_ctx = f"- Date: {state.get('email_date')}\n" + email_ctx
                system_prompt += f"\n\nEMAIL CONTEXT:\n{email_ctx}" + client_ctx
            else:
                system_prompt += client_ctx

            if state.get("flag_create_missing"):
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
                            
                            import re
                            def clean_acct(s):
                                return re.sub(r'[\u2600-\u27BF\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2300-\u23FF\u2B00-\u2BFF\uFE00-\uFE0F]+', '', s).strip()
                            all_account_names = [clean_acct(acc.get("name")) for acc in accounts if acc.get("name")]
                            all_accounts_json = json.dumps(all_account_names).replace("{", "{{").replace("}", "}}")
                            system_prompt += f"\n\nTRANSFER ACCOUNTS:\nIf a proposed transaction is actually a transfer of money to another account (e.g., an e-transfer to yourself is a transfer to another chequing account, an Uber redemption is a transfer to the Credits account), omit the 'category' field and instead provide 'transfer_account' matching EXACTLY one of these account names: {all_accounts_json}"
                        except Exception as json_e:
                            print(f"[parse_receipt][error]   ⚠️ Failed to parse accounts JSON: {json_e}")
                    
                    categories_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../actual-utils/cache/categories.json"))
                    if os.path.exists(categories_path):
                        with open(categories_path, "r") as f:
                            categories = json.load(f)
                        if isinstance(categories, list):
                            expense_categories = [c.get("name") for c in categories if isinstance(c, dict) and not c.get("hidden") and not c.get("is_income") and c.get("name")]
                            income_categories = [c.get("name") for c in categories if isinstance(c, dict) and not c.get("hidden") and c.get("is_income") and c.get("name")]
                            if expense_categories:
                                expense_json = json.dumps(expense_categories).replace("{", "{{").replace("}", "}}")
                                income_json = json.dumps(income_categories).replace("{", "{{").replace("}", "}}") if income_categories else "[]"
                                system_prompt += f"\n\nCATEGORY SELECTION:\nEach proposed_transaction must have a 'category' (unless it's a transfer). Pick from these expense categories: {expense_json}\nFor deposits, refunds, credits, or points redemptions (negative amounts), use one of these income categories instead: {income_json}\nIf unsure, pick the most generic expense category."
                except Exception as e:
                    print(f"[parse_receipt][error]   ⚠️ Failed to load on-budget accounts for prompt: {e}")

            if current_attempt.get("append_prompt"):
                prev_date = state.get("parsed_data", {}).get("date", "unknown")
                prompt_addition = current_attempt["append_prompt"].replace("{prev_date}", str(prev_date))
                system_prompt += prompt_addition

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Here is the {source_desc}:\n\n{ocr_text}")
            ])
            
            todays_date = datetime.date.today().isoformat()
            prompt_val = prompt.format_prompt(source_desc=source_desc, ocr_text=state["ocr_text"], todays_date=todays_date)
            print(f"[parse_receipt][agent]  Extract prompt: ~{estimate_tokens(prompt_val.to_string())} tokens")
            
            if state.get("flag_log_agent"):
                print(f"\n[parse_receipt][agent] --- AGENT PROMPT ({model_name}) ---")
                print(f"[parse_receipt] {prompt_val.to_string()}")
                print("[parse_receipt] " + "---------------------------------------\n")

            try:
                import time
                start_time = time.time()
                structured_llm = llm.with_structured_output(ParsedReceipt, method="json_mode", include_raw=True)
                chain = prompt | structured_llm
                result_wrapper = await chain.ainvoke({"ocr_text": state["ocr_text"], "source_desc": source_desc, "todays_date": todays_date})
                response_time_ms = int((time.time() - start_time) * 1000)
                
                result = result_wrapper.get("parsed")
                raw_msg = result_wrapper.get("raw")
                
                prompt_tokens = None
                completion_tokens = None
                if raw_msg and hasattr(raw_msg, "usage_metadata") and raw_msg.usage_metadata:
                    prompt_tokens = raw_msg.usage_metadata.get("input_tokens")
                    completion_tokens = raw_msg.usage_metadata.get("output_tokens")
                
                parsed_dict = result.dict()
                
                if state.get("flag_log_agent"):
                    print(f"\n[parse_receipt][agent] --- AGENT RESPONSE ({model_name}) ---")
                    print(f"[parse_receipt] {json.dumps(parsed_dict, indent=2)}")
                    print(f"[parse_receipt] [Metrics] Time: {response_time_ms}ms, Input Tokens: {prompt_tokens}, Output Tokens: {completion_tokens}")
                    print("[parse_receipt] " + "---------------------------------------\n")
            except Exception as e:
                error_msg = str(e)
                try:
                    cause = getattr(e, "__cause__", None) or e
                    if hasattr(cause, "errors") and callable(getattr(cause, "errors")):
                        fields = []
                        for err in cause.errors():
                            loc = ".".join(str(x) for x in err.get("loc", []))
                            fields.append(f"'{loc}'")
                        if fields:
                            error_msg = f"Validation failed on fields: {', '.join(fields)}"
                    elif "validation error" in error_msg.lower():
                        lines = error_msg.split('\n')
                        for i, line in enumerate(lines):
                            if "validation error" in line.lower() and i + 1 < len(lines):
                                field = lines[i+1].strip()
                                error_msg = f"Validation failed on field '{field}'"
                                break
                except Exception:
                    pass

                print(f"[parse_receipt][error]   ❌ Parsing failed: {error_msg}. Flagging for review.")
                llm_out = getattr(e, "llm_output", None)
                if not llm_out and "from completion " in str(e):
                    import re
                    match = re.search(r'from completion (\{.*?\})\. Got:', str(e), re.DOTALL)
                    if match:
                        llm_out = match.group(1)
                
                if llm_out:
                    if not isinstance(llm_out, str):
                        llm_out = str(llm_out)
                    try:
                        parsed_out = json.loads(llm_out)
                        llm_out_fmt = json.dumps(parsed_out, indent=2)
                        print(f"[parse_receipt][agent]   Agent response was:\n{llm_out_fmt}")
                    except Exception:
                        print(f"[parse_receipt][agent]   Agent response was:\n{llm_out}")

                agent_attempts.append({
                    "node": "receipt_parse", 
                    "model": model_name, 
                    "success": False, 
                    "error": error_msg,
                    "prompt_tokens": prompt_tokens if 'prompt_tokens' in locals() else None,
                    "completion_tokens": completion_tokens if 'completion_tokens' in locals() else None,
                    "response_time_ms": response_time_ms if 'response_time_ms' in locals() else None
                })
                return {"agent_attempts": agent_attempts}
        
        # Email date override logic
        from actual_orch.shared.utils import get_receipt_type
        if get_receipt_type(state.get("source")) == "email" and state.get("email_date"):
            if parse_retries == 0:
                try:
                    email_dt = datetime.datetime.strptime(state["email_date"], "%Y-%m-%d")
                    if parsed_dict.get("date"):
                        parsed_dt = datetime.datetime.strptime(parsed_dict["date"], "%Y-%m-%d")
                        if (email_dt - parsed_dt).days > 3:
                            print(f"[parse_receipt]  Overriding parsed date {parsed_dict['date']} with email date {state['email_date']} (>3 days older)")
                            parsed_dict["date"] = state["email_date"]
                    else:
                        parsed_dict["date"] = state["email_date"]
                except Exception as e:
                    print(f"[parse_receipt][error]   ⚠️ Could not compare dates for override: {e}")
        
        # Validate proposed_transactions — each must have amount, notes, category_id
        parsed_pts = parsed_dict.get("proposed_transactions") or []
        
        # Backward compat: v1 cached data may lack proposed_transactions
        if not parsed_pts and parsed_dict.get("total"):
            parsed_pts = [{
                "amount": parsed_dict["total"],
                "notes": parsed_dict.get("notes", ""),
                "category_id": parsed_dict.get("category_id", "")
            }]
            parsed_dict["proposed_transactions"] = parsed_pts
        
        for i, pt in enumerate(parsed_pts):
            missing = [f for f in ("amount", "notes") if not pt.get(f) and pt.get(f) != 0]
            if missing:
                print(f"[parse_receipt][error]   ⚠️ proposed_transaction[{i}] missing fields: {missing} — {pt}")
            if not pt.get("category") and not pt.get("transfer_account"):
                print(f"[parse_receipt][error]   ⚠️ proposed_transaction[{i}] missing both category and transfer_account — {pt}")
        
        # Calculate total from sum of proposed transaction amounts (in receipt currency)
        receipt_total = sum(pt.get("amount", 0.0) for pt in parsed_pts)
        
        # Currency / FX handling — skip if total already computed (cached data)
        if "total" not in parsed_dict or parsed_dict["total"] == 0:
            currency = (parsed_dict.get("currency") or "CAD").upper()
            parsed_dict["currency"] = currency
            forex_receipt = currency != "CAD"
            
            if forex_receipt and receipt_total:
                parsed_dict["foreign_amount"] = receipt_total
                try:
                    import sys
                    utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../actual-utils"))
                    if utils_path not in sys.path:
                        sys.path.append(utils_path)
                    from modules.market import get_price
                    
                    price, _ = get_price(currency, parsed_dict.get("date"))
                    if price:
                        parsed_dict["total"] = round(receipt_total * price, 2)
                        print(f"[parse_receipt]  Converted {receipt_total} {currency} to approx {parsed_dict['total']} CAD")
                    else:
                        parsed_dict["total"] = receipt_total
                except Exception as e:
                    print(f"[parse_receipt][error]   ⚠️ Could not convert foreign amount to CAD: {e}")
                    parsed_dict["total"] = receipt_total
            else:
                parsed_dict["total"] = receipt_total
                
        # Cache the raw agent response (only for fresh LLM calls, not cached data)
        if model_name:
            meta = {
                "response_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "model": model_name,
                "schema_version": "2.0.0"
            }
            set_parsed(file_path, parsed_dict, meta, prefix=prefix)
        
        config_path_class = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/classification.json"))
        skip_if_future_date = False
        classification = state.get("classification", "")
        if os.path.exists(config_path_class):
            with open(config_path_class, "r") as f:
                class_data = json.load(f)
                for cls_name, cls_config in class_data.items():
                    if cls_name == classification.strip():
                        skip_if_future_date = cls_config.get("skip_if_future_date", False)
                        break

        if parsed_dict.get("date"):
            try:
                parsed_date = datetime.datetime.strptime(parsed_dict["date"], "%Y-%m-%d").date()
                if datetime.date.today() < parsed_date:
                    from actual_orch.shared.utils import get_receipt_type
                    if get_receipt_type(state.get("source")) == "email" and state.get("email_date"):
                        print(f"[parse_receipt][error]   ⚠️ Parsed date {parsed_date} is in the future. Falling back to email date {state['email_date']}.")
                        parsed_dict["date"] = state["email_date"]
                    elif skip_if_future_date:
                        print(f"[parse_receipt][skip]   ⏭️  Parsed date {parsed_date} is in the future. Skipping.")
                        return {"status": WorkflowStatus.ALREADY_PROCESSED, "parsed_data": parsed_dict, "agent_attempts": agent_attempts}
            except Exception:
                pass
            
        if model_name:
            agent_attempts.append({
                "node": "receipt_parse", 
                "model": model_name, 
                "success": True, 
                "error": None,
                "prompt_tokens": prompt_tokens if 'prompt_tokens' in locals() else None,
                "completion_tokens": completion_tokens if 'completion_tokens' in locals() else None,
                "response_time_ms": response_time_ms if 'response_time_ms' in locals() else None
            })
        
        # Build proposed_transactions for state (resolve display names, copy date/payee/account)
        proposed_list = []
            
        # Resolve account_key -> account name
        account_name = parsed_dict.get("account_key")
        account_key = parsed_dict.get("account_key")
        if account_key and state.get("flag_create_missing"):
            try:
                mapping_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../actual-utils/cache/account-mapping.json"))
                if os.path.exists(mapping_path):
                    with open(mapping_path, "r") as f:
                        acct_mapping = json.load(f)
                    account_id = acct_mapping.get(account_key)
                    if account_id:
                        from actual_orch.clients.mcp import ActualMCPClient
                        acct_session = await ActualMCPClient.get_session()
                        acct_res = await acct_session.call_tool("actual_accounts", {"args": ["list", "--format", "json"]})
                        if acct_res.content:
                            accts = json.loads(acct_res.content[0].text)
                            acct_match = next((a for a in accts if a.get("id") == account_id), None)
                            if acct_match:
                                account_name = acct_match.get("name", account_key)
            except Exception as e:
                print(f"[parse_receipt][error]   ⚠️ Failed to resolve account name: {e}")
                
        for pt in parsed_pts:
            p = {
                "date": parsed_dict.get("date", ""),
                "payee": parsed_dict.get("payee", ""),
                "notes": pt.get("notes", ""),
                "amount": pt.get("amount", 0.0)
            }
            if account_name:
                p["account"] = account_name
                
            cat = pt.get("category")
            if cat:
                p["category"] = cat
                
            transfer = pt.get("transfer_account")
            if transfer:
                p["transfer_account"] = transfer
                
            proposed_list.append(p)
            
        return {"parsed_data": parsed_dict, "agent_attempts": agent_attempts, "proposed_transactions": proposed_list}
    except Exception as e:
        return {"status": WorkflowStatus.ERROR, "error": f"Parsing failed: {str(e)}"}


def human_approval_node(state: ReceiptState):
    """Placeholder node for human interrupt"""
    return {}

def router_after_dedup(state: ReceiptState):
    if state.get("status") == WorkflowStatus.ALREADY_PROCESSED:
        return "skip"
    return "continue"

def router_check_error(state: ReceiptState):
    if state.get("status") == WorkflowStatus.ERROR:
        return "error"
    return "continue"

def router_after_classify(state: ReceiptState):
    if state.get("status") == WorkflowStatus.ERROR:
        return "error"
    if state.get("status") == WorkflowStatus.ALREADY_PROCESSED:
        return "skip"
        
    classification = state.get("classification", "")
    config_path_class = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/classification.json"))
    should_parse = True
    
    if os.path.exists(config_path_class):
        with open(config_path_class, "r") as f:
            class_data = json.load(f)
            for cls_name, cls_config in class_data.items():
                if cls_name == classification.strip():
                    should_parse = cls_config.get("parse", True)
                    break
            
    if not should_parse:
        return "skip"
    return "continue"

def router_after_parse(state: ReceiptState):
    if state.get("status") == WorkflowStatus.ERROR:
        return "error"
    if state.get("status") == WorkflowStatus.ALREADY_PROCESSED:
        return "skip"
        
    attempts = [a for a in state.get("agent_attempts", []) if a.get("node") == "receipt_parse"]
    if attempts and not attempts[-1].get("success", False):
        return "human_approval"
        
    classification = state.get("classification", "")
    config_path_class = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/classification.json"))
    skip_if_no_amount = False
    
    if os.path.exists(config_path_class):
        with open(config_path_class, "r") as f:
            class_data = json.load(f)
            for cls_name, cls_config in class_data.items():
                if cls_name == classification.strip():
                    skip_if_no_amount = cls_config.get("skip_if_no_amount", False)
                    break
                    
    if skip_if_no_amount:
        parsed_data = state.get("parsed_data", {})
        total = parsed_data.get("total", 0)
        if not total and total != 0.0:
            total = 0
        if float(total) == 0.0:
            print(f"[router_after_parse][skip]   ⏭️  Skipped {state.get('source_ref')}: Total amount is 0. (Classified as: {classification})")
            return "skip"
            
    return "continue"

def router_after_search_deterministic(state: ReceiptState):
    if state.get("status") == WorkflowStatus.ERROR:
        return "error"
    if state.get("status") == WorkflowStatus.FALLBACK_TO_AGENT:
        return "fallback_to_agent"
    if state.get("actual_tx_id") != "unknown_id":
        return "update_actual"
    return "error"

def router_after_search_agent(state: ReceiptState):
    if state.get("status") == WorkflowStatus.ERROR:
        return "error"
    if state.get("status") == WorkflowStatus.RETRY_PARSE:
        return "retry_parse"
    if state.get("status") == WorkflowStatus.HUMAN_APPROVAL or state.get("actual_tx_id") == "unknown_id":
        if state.get("flag_create_missing"):
            return "save_actual"
        return "human_approval"
    return "update_actual"

def router_after_human(state: ReceiptState):
    if state.get("flag_create_missing") and state.get("actual_tx_id") == "unknown_id":
        return "save_actual"
    if state.get("actual_tx_id") and state.get("actual_tx_id") != "unknown_id":
        return "update_actual"
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
workflow.add_node("update_actual", update_actual_transaction_node)
workflow.add_node("save_pocketbase", save_pocketbase_node)

workflow.set_entry_point("skip_if_processed")

workflow.add_conditional_edges("skip_if_processed", router_after_dedup, {"skip": END, "continue": "extract_ocr"})
workflow.add_conditional_edges("extract_ocr", router_check_error, {"error": END, "continue": "classify_input"})
workflow.add_conditional_edges("classify_input", router_after_classify, {"error": END, "human_approval": "human_approval", "continue": "parse_receipt", "skip": END})
workflow.add_conditional_edges("parse_receipt", router_after_parse, {"error": END, "human_approval": "human_approval", "continue": "search_deterministic", "skip": END})
workflow.add_conditional_edges("search_deterministic", router_after_search_deterministic, {
    "error": END, 
    "fallback_to_agent": "search_agent", 
    "update_actual": "update_actual"
})
workflow.add_conditional_edges("search_agent", router_after_search_agent, {
    "error": END, 
    "human_approval": "human_approval", 
    "save_actual": "save_actual",
    "update_actual": "update_actual",
    "retry_parse": "parse_receipt"
})
workflow.add_conditional_edges("human_approval", router_after_human, {"save_actual": "save_actual", "update_actual": "update_actual", "skip": END})
workflow.add_edge("save_actual", "update_actual")
workflow.add_edge("update_actual", "save_pocketbase")
workflow.add_edge("save_pocketbase", END)
