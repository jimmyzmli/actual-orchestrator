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

from actual_orch.clients.cache import get_ocr, set_ocr, get_parsed, set_parsed
import datetime

class StatementState(TypedDict, total=False):
    file_path: str
    ocr_text: str
    parsed_data: Dict[str, Any]
    status: str
    error: str
    image_paths: List[str]
    use_gemini_only: bool
    ocr_tool: str

class ParsedInvestmentTx(BaseModel):
    date: Any = Field(description="Date of the transaction in YYYY-MM-DD format", default="")
    type: Any = Field(description="Type of the transaction. Must be one of: Buy, Sell, Dividend, Interest, Contribution, Transfer, Withdrawal, Deposit", default="")
    symbol: Any = Field(description="Stock ticker symbol or identifier. Use empty string if none.", default="")
    quantity: Any = Field(description="Number of shares or units. Use 0.0 if not applicable (e.g. cash deposit)", default=0.0)
    price: Any = Field(description="Price per share or unit. Use 0.0 if not applicable", default=0.0)
    amount: Any = Field(description="Total monetary amount of the transaction. Use 0.0 if none", default=0.0)
    currency: Any = Field(description="Currency code (e.g. CAD, USD)", default="CAD")
    description: Any = Field(description="Raw or summarized description of the transaction from the statement", default="")

class ParsedStatement(BaseModel):
    transactions: List[ParsedInvestmentTx] = Field(description="List of extracted investment transactions")


def extract_ocr_node(state: StatementState):
    """Extract text from the PDF using local Docling."""
    try:
        file_path = state["file_path"]

        image_paths = [] # Not needed for docling, but kept for schema compatibility

        cached = get_ocr(file_path, prefix="statement_")
        if cached is not None:
            print("[extract_ocr][cache] " + "Using cached OCR text...")
            return {"ocr_text": cached, "image_paths": image_paths}

        ocr_tool = state.get("ocr_tool", "docling")
        print(f"[extract_ocr] Extracting text with {ocr_tool} for {file_path}...")
        
        if ocr_tool == "pdfplumber":
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                ocr_text = "\n\n".join([page.extract_text() or "" for page in pdf.pages])
        else:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(file_path)
            ocr_text = result.document.export_to_markdown()
            
        if not ocr_text or not ocr_text.strip():
            print(f"[extract_ocr]   -> WARNING: OCR text is empty!")
            
        meta = {
            "source_file": f"file://{os.path.abspath(file_path)}",
            "ocr_generation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "ocr_model_used": ocr_tool,
            "response_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": ocr_tool,
            "schema_version": "1.0.0"
        }
        set_ocr(file_path, ocr_text, meta, prefix="statement_")
            
        return {"ocr_text": ocr_text, "image_paths": image_paths}
    except Exception as e:
        return {"status": "error", "error": f"OCR failed: {str(e)}"}


def parse_statement_node(state: StatementState):
    """Parse the OCR text using DeepSeek-V4-Pro."""
    try:
        if state.get("status") == "error":
            return {} # Preserve existing error

        if not state.get("ocr_text"):
            return {"status": "error", "error": "No OCR text found"}

        file_path = state["file_path"]

        cached = get_parsed(file_path, prefix="statement_")
        if cached is not None:
            print("[parse_statement][cache] " + "Using cached parsed statement data...")
            return {"parsed_data": cached}

        print(f"[parse_statement] Parsing statement with DeepSeek-V4-Pro...")
        
        from actual_orch.shared.utils import get_agent_config
        attempts = get_agent_config("statement_parse").get("attempts", [])
        model_name = attempts[0].get("model", "deepseek-chat") if attempts else "deepseek-chat"

        llm = ChatOpenAI(
            model=model_name, # API identifier for DeepSeek-V4-Pro
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""), 
            base_url="https://api.deepseek.com/v1",
            max_tokens=8000,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert financial data extractor. Your task is to extract all investment transactions from the markdown text of a brokerage statement. Focus on Buys, Sells, Dividends, Interest, Contributions, Transfers, Withdrawals, and Deposits. Be precise with Dates (YYYY-MM-DD), Quantities, Prices, and Amounts. Match the types strictly. NEVER output null for any fields. If a number is missing or not applicable, output 0.0. If text is missing, output an empty string.\nCRITICAL LAYOUT RULES:\n1. Currency: Pay close attention to page-level or section-level currency headers (e.g., 'TFSA Statement (U.S.$)', 'TFSA Statement (CDN$)'). Apply the correct currency to all transactions on that page/section.\n2. Multi-line Symbols: In some statements (e.g. RBC), a transaction block may have the company name and symbol on the first line (e.g. 'BOUGHT BETTERWAREDEMEXICO 122 43.689 5,340.02') and the activity (e.g. 'UNSOLICITED DA') on subsequent lines. Combine them to correctly identify the symbol and description, and do not drop the symbol.\nRespond with a JSON object containing a 'transactions' array."),
            ("human", "Here is the markdown text of the statement:\n\n{ocr_text}")
        ])
        
        structured_llm = llm.with_structured_output(ParsedStatement, method="json_mode")
        chain = prompt | structured_llm
        
        try:
            from langchain_community.callbacks.manager import get_openai_callback
            with get_openai_callback() as cb:
                result = chain.invoke({"ocr_text": state["ocr_text"]})
                print(f"[parse_statement]   -> DeepSeek Tokens Used: {cb.total_tokens} (Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens})")
        except ImportError:
            result = chain.invoke({"ocr_text": state["ocr_text"]})
            print("[parse_statement] " + "  -> DeepSeek parsing completed (token usage unavailable without langchain-community)")
        
        parsed_dict = result.dict()
        meta = {
            "source_file": f"file://{os.path.abspath(file_path)}",
            "parse_generation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "parse_model_used": "deepseek-chat",
            "schema_version": "1.0.0",
            "response_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": "deepseek-chat"
        }
        set_parsed(file_path, parsed_dict, meta, prefix="statement_")
            
        return {"parsed_data": parsed_dict}
    except Exception as e:
        return {"status": "error", "error": f"Parsing failed: {str(e)}"}


def extract_and_parse_statement_node(state: StatementState):
    """Alternative holistic path: Extract and parse simultaneously using Gemini only."""
    try:
        file_path = state["file_path"]

        cached = get_parsed(file_path)
        if cached is not None:
            print("[extract_and_parse_statement][cache] " + "Using cached parsed statement data...")
            return {"parsed_data": cached}

        image_paths = state.get("image_paths", [])
        if not image_paths:
            if file_path.lower().endswith(".pdf"):
                gen_dir = os.path.join(os.getcwd(), "cache", "generated_image")
                os.makedirs(gen_dir, exist_ok=True)
                images = convert_from_path(file_path, dpi=200)
                for idx, img in enumerate(images):
                    safe_name = os.path.basename(file_path) + f"_page_{idx}.jpg"
                    img_path = os.path.join(gen_dir, safe_name)
                    if not os.path.exists(img_path):
                        img.save(img_path, format="JPEG")
                    image_paths.append(img_path)
            else:
                image_paths.append(file_path)

        print(f"[extract_and_parse_statement] Holistic extraction and parsing with Gemini for {file_path}...")
        
        import time
        time.sleep(4.1) # Respect Gemini free tier limits
        
        from actual_orch.shared.utils import get_agent_config
        attempts = get_agent_config("statement_holistic").get("attempts", [])
        model_name = attempts[0].get("model", "gemini-3.1-pro") if attempts else "gemini-3.1-pro"

        llm = ChatGoogleGenerativeAI(
            model=model_name, # Use the pro model for complex extraction
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            temperature=0
        )
        
        structured_llm = llm.with_structured_output(ParsedStatement)
        
        content = [
            {"type": "text", "text": "You are an expert financial data extractor. Extract all investment transactions from these pages of a brokerage statement. Focus on Buys, Sells, Dividends, Interest, Contributions, and Transfers. Be precise with Dates (YYYY-MM-DD), Quantities, Prices, Amounts, and Currencies. Output the results strictly adhering to the schema."}
        ]
        
        for img_path in image_paths:
            with open(img_path, "rb") as f:
                img_str = base64.b64encode(f.read()).decode("utf-8")
            
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
            })
            
        messages = [HumanMessage(content=content)]
        result = structured_llm.invoke(messages)
        
        parsed_dict = result.dict()
        meta = {
            "response_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": "gemini-3.1-pro",
            "schema_version": "1.0.0"
        }
        set_parsed(file_path, parsed_dict, meta)
            
        return {"parsed_data": parsed_dict, "image_paths": image_paths}
    except Exception as e:
        return {"status": "error", "error": f"Holistic parsing failed: {str(e)}"}


def route_start(state: StatementState):
    if state.get("use_gemini_only"):
        return "extract_and_parse_statement"
    return "extract_ocr"

# Build Graph
workflow = StateGraph(StatementState)

workflow.add_node("extract_ocr", extract_ocr_node)
workflow.add_node("parse_statement", parse_statement_node)
workflow.add_node("extract_and_parse_statement", extract_and_parse_statement_node)

workflow.set_conditional_entry_point(route_start, {
    "extract_ocr": "extract_ocr",
    "extract_and_parse_statement": "extract_and_parse_statement"
})

workflow.add_edge("extract_ocr", "parse_statement")
workflow.add_edge("parse_statement", END)
workflow.add_edge("extract_and_parse_statement", END)

graph_builder = workflow
app = graph_builder.compile()
# For backward compatibility with clients expecting 'workflow' to be the runnable
workflow = app

