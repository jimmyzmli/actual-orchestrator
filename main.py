import os
import sys
import json
import subprocess
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from graph.workflow import receipt_processor

def fetch_unprocessed_receipts(gdrive_path: str):
    """
    Use rclone to list files in the gdrive path.
    Returns a list of dicts with file info.
    """
    print(f"Fetching files from {gdrive_path}...")
    try:
        # rclone lsjson gdrive:Receipts
        result = subprocess.check_output(["rclone", "lsjson", gdrive_path], text=True)
        files = json.loads(result)
        # Filter for files (not dirs) and PDFs/Images
        receipt_files = [f for f in files if not f.get("IsDir") and f.get("Name", "").lower().endswith((".pdf", ".jpg", ".jpeg", ".png"))]
        return receipt_files
    except subprocess.CalledProcessError as e:
        print(f"Error fetching from rclone: {e}")
        return []
    except FileNotFoundError:
        print("rclone is not installed or not in PATH")
        return []

def download_file(gdrive_path: str, filename: str, dest_dir: str):
    """
    Download a file from gdrive using rclone to a local directory.
    Returns the local file path.
    """
    source_path = f"{gdrive_path}/{filename}"
    print(f"Downloading {source_path} to {dest_dir}...")
    subprocess.check_call(["rclone", "copy", source_path, dest_dir])
    return os.path.join(dest_dir, filename)

def process_receipt(local_path: str, gdrive_source_ref: str):
    """
    Run the LangGraph workflow for a single receipt.
    """
    import asyncio
    
    state = {
        "file_path": local_path,
        "gdrive_path": gdrive_source_ref,
        "status": "pending"
    }
    
    # We must use a unique thread_id for checkpointers
    config = {"configurable": {"thread_id": gdrive_source_ref}}
    
    # Run graph
    print(f"\n--- Starting processing for {gdrive_source_ref} ---")
    
    async def run_flow():
        # Step 1: Initial invocation
        result = await receipt_processor.ainvoke(state, config)
        
        # Step 2: Check if interrupted
        current_state = receipt_processor.get_state(config)
        if "human_approval" in current_state.next:
            print(f"⚠️ Transaction needs manual review. Pausing and deferring to end of batch.")
            return {"status": "paused", "config": config, "state": current_state}
            
        return {"status": "finished"}

    return asyncio.run(run_flow())

def main():
    gdrive_path = os.environ.get("RECEIPT_GDRIVE_PATH")
    if not gdrive_path:
        print("RECEIPT_GDRIVE_PATH not set in .env")
        sys.exit(1)
        
    receipts = fetch_unprocessed_receipts(gdrive_path)
    if not receipts:
        print("No receipts found or error fetching.")
        return
        
    print(f"Found {len(receipts)} receipts to process.")
    # Use a persistent local directory for testing and caching
    tmpdir = os.path.join(os.getcwd(), "cache")
    os.makedirs(tmpdir, exist_ok=True)

    # Batch-fetch all known source_hashes from PocketBase in one sweep
    # so the per-file dedup check is an O(1) set lookup, not an HTTP call.
    from tools.pocketbase_client import fetch_all_source_hashes
    known_hashes = fetch_all_source_hashes()
    print(f"Loaded {len(known_hashes)} existing receipt hashes from PocketBase.")
    
    paused_receipts = []
    
    for receipt in receipts:
        filename = receipt["Name"]
        gdrive_source_ref = f"{gdrive_path}/{filename}"
        
        try:
            # Download if needed (cheap — rclone skips if already present)
            local_path = os.path.join(tmpdir, filename)
            if not os.path.exists(local_path):
                local_path = download_file(gdrive_path, filename, tmpdir)
            else:
                print(f"Using already downloaded file: {local_path}")

            # Dedup by file content hash (O(1) set lookup, no HTTP)
            from graph.cache import _file_hash
            file_hash = _file_hash(local_path)
            if file_hash in known_hashes:
                print(f"Skipping {gdrive_source_ref}: already processed in PocketBase (hash {file_hash[:12]}...).")
                continue

            res = process_receipt(local_path, gdrive_source_ref)
            if res and res.get("status") == "paused":
                paused_receipts.append((gdrive_source_ref, local_path, res["config"], res["state"]))
                
            # Optionally, move the file to a 'Processed' folder in gdrive
            # subprocess.check_call(["rclone", "move", gdrive_source_ref, f"{gdrive_path}/Processed/"])
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    if paused_receipts:
        import asyncio
        print(f"\n\n{'='*50}\nINTERACTIVE REVIEW PHASE ({len(paused_receipts)} receipts)\n{'='*50}")
        for ref, local_path, cfg, st in paused_receipts:
            print(f"\nReviewing: {ref}")
            import urllib.parse
            encoded_path = urllib.parse.quote(local_path)
            print(f"File: 'file://{encoded_path}'")
            
            parsed = st.values.get("parsed_data")
            if parsed:
                print(f"Parsed Date:   {parsed.get('date')}")
                print(f"Parsed Payee:  {parsed.get('payee')}")
                print(f"Parsed Amount: ${parsed.get('total')}")
            
            proposed = st.values.get("proposed_transaction")
            if proposed:
                print(f"⚠️ Transaction NOT found or amount mismatched. Agent proposes:")
                print(f"{proposed}")
                ans = input("Do you want to create this transaction? [y/N] (or enter UUID to override): ").strip().lower()
            else:
                print(f"⚠️ Agent failed to match or propose a transaction.")
                ans = input("Enter a valid transaction UUID to override, or 'n' to skip: ").strip().lower()
            
            import re
            if re.match(r'^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$', ans, re.IGNORECASE):
                asyncio.run(receipt_processor.aupdate_state(cfg, {"actual_tx_id": ans, "human_approved": False}))
            else:
                approved = bool(proposed and ans in ("y", "yes"))
                asyncio.run(receipt_processor.aupdate_state(cfg, {"human_approved": approved}))
            
            print(f"Resuming workflow for {ref}...")
            asyncio.run(receipt_processor.ainvoke(None, cfg))
            print("Done.\n")
    else:
        print("\nAll receipts processed successfully! No manual reviews needed.")
        
    import asyncio
    asyncio.run(run_final_validation())

async def run_final_validation():
    print(f"\n\n{'='*50}\nFINAL VALIDATION\n{'='*50}")
    # 1. Pull PB tx ids
    pb_tx_ids = set()
    from tools.pocketbase_client import get_pb_client
    try:
        client = get_pb_client()
        records = client.collection("receipts").get_full_list(query_params={"fields": "tx_id"})
        for r in records:
            if hasattr(r, 'tx_id') and r.tx_id and r.tx_id != "unknown_id":
                pb_tx_ids.add(r.tx_id)
    except Exception as e:
        print(f"Failed to pull from PocketBase: {e}")
        return

    # 2. Pull Actual Budget tx ids
    actual_tx_ids = set()
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import os, json
    
    server_params = StdioServerParameters(
        command="node",
        args=["wrapper.js"],
        env=os.environ.copy()
    )
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool("actual_query", {"args": [
                    "run", "--table", "transactions", 
                    "--filter", json.dumps({"notes": {"$like": "%#receipt%"}}), 
                    "--select", "id", "--format", "json"
                ]})
                if res.content and len(res.content) > 0 and hasattr(res.content[0], 'text'):
                    data = json.loads(res.content[0].text)
                    if isinstance(data, list):
                        for tx in data:
                            actual_tx_ids.add(tx.get("id"))
    except Exception as e:
        print(f"Failed to pull from Actual Budget: {e}")
        return

    # 3. Find mismatches
    in_pb_not_actual = pb_tx_ids - actual_tx_ids
    in_actual_not_pb = actual_tx_ids - pb_tx_ids
    
    if not in_pb_not_actual and not in_actual_not_pb:
        print("✅ Validation passed. All receipts perfectly synced.")
    else:
        if in_pb_not_actual:
            print(f"⚠️ {len(in_pb_not_actual)} receipts in PocketBase are MISSING the #receipt tag in Actual Budget:")
            for tx in in_pb_not_actual:
                print(f"  - {tx}")
        if in_actual_not_pb:
            print(f"⚠️ {len(in_actual_not_pb)} transactions in Actual Budget have #receipt tag but are MISSING in PocketBase:")
            for tx in in_actual_not_pb:
                print(f"  - {tx}")

if __name__ == "__main__":
    main()
