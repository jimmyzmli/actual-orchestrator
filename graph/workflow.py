import os
import json
import base64
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from pdf2image import convert_from_path

from tools.actual_mcp import find_or_create_transaction
from tools.pocketbase_client import save_receipt
from graph.cache import get_ocr, set_ocr, get_parsed, set_parsed

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

async def search_actual_transaction_node(state: ReceiptState):
    """Search for the transaction in Actual Budget using an Agent"""
    final_id = "unknown_id"
    proposed_tx = None
    
    try:
        print("Matching transaction in Actual Budget via Agent...")
        parsed = state.get("parsed_data")
        if not parsed:
            return {"status": "error", "error": "No parsed data found"}
            
        date = parsed["date"]
        amount = parsed["total"]
        payee = parsed["payee"]
        
        notes = "Extracted from receipt. Items:\n"
        for item in parsed.get("items", []):
            notes += f"- {item['description']}: ${item['amount']}\n"
            
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
                
                # Only give it search tools!
                search_tools = [t for t in mcp_tools if "actual_query" in t.name or "actual_accounts" in t.name]
                agent = create_react_agent(llm, tools=search_tools)
                
                prompt = f"""
                You are a financial assistant connecting to Actual Budget.
                We have a new parsed receipt:
                - Date: {date}
                - Amount: ${amount}
                - Payee: {payee}
                - Notes to append: {notes}
                
                Task:
                1. Use actual_query to search transactions within +/- 1 day of {date} for this exact amount (convert to cents, expenses are negative!).
                2. If found, return its ID. If multiple match, pick the first one.
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
                    
                    # STRICT AMOUNT VALIDATION
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
                                    if abs(actual_amt) != abs(expected_amt):
                                        print(f"❌ STRICT VALIDATION FAILED: Actual Budget amount ${abs(actual_amt)/100:.2f} != parsed receipt amount ${amount:.2f}. Rejecting match.")
                                        final_id = "unknown_id"
                        except Exception as e:
                            print(f"Warning: Failed to strictly validate amount: {e}")
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
        
        res = save_receipt(tx_id, lines, date, source_ref, amount, payee, image_paths)
        
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "error": f"PocketBase save failed: {str(e)}"}

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

workflow.add_node("extract_ocr", extract_ocr_node)
workflow.add_node("parse_receipt", parse_receipt_node)
workflow.add_node("search_actual", search_actual_transaction_node)
workflow.add_node("human_approval", human_approval_node)
workflow.add_node("create_actual", create_actual_transaction_node)
workflow.add_node("save_pocketbase", save_pocketbase_node)

workflow.set_entry_point("extract_ocr")

workflow.add_conditional_edges("extract_ocr", router_check_error, {"error": END, "continue": "parse_receipt"})
workflow.add_conditional_edges("parse_receipt", router_check_error, {"error": END, "continue": "search_actual"})
workflow.add_conditional_edges("search_actual", router_after_search, {"error": END, "human_approval": "human_approval", "save_pocketbase": "save_pocketbase"})
workflow.add_conditional_edges("human_approval", router_after_human, {"create_actual": "create_actual", "save_pocketbase": "save_pocketbase", "skip": END})
workflow.add_conditional_edges("create_actual", router_check_error, {"error": END, "continue": "save_pocketbase"})
workflow.add_edge("save_pocketbase", END)

memory = MemorySaver()
receipt_processor = workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])
