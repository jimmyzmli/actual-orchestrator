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
        # Filter for files (not dirs) and PDFs
        pdfs = [f for f in files if not f.get("IsDir") and f.get("Name", "").lower().endswith(".pdf")]
        return pdfs
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
    
    paused_receipts = []
    
    for receipt in receipts:
        filename = receipt["Name"]
        gdrive_source_ref = f"{gdrive_path}/{filename}"
        
        try:
            from tools.pocketbase_client import check_receipt_exists
            if check_receipt_exists(gdrive_source_ref):
                print(f"Skipping {gdrive_source_ref}: already processed in PocketBase.")
                continue

            # We can also check if the PDF is already downloaded to save time
            local_path = os.path.join(tmpdir, filename)
            if not os.path.exists(local_path):
                local_path = download_file(gdrive_path, filename, tmpdir)
            else:
                print(f"Using already downloaded file: {local_path}")
                
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

if __name__ == "__main__":
    main()
