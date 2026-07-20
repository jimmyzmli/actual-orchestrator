import os
import sys
import json
import subprocess
import tempfile
import argparse
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
            
            not_a_receipt = st.values.get("not_a_receipt", False)
            
            if not_a_receipt:
                print(f"🚫 Content was classified as NOT a receipt.")
                print(f"   OCR preview: {(st.values.get('ocr_text') or '')[:200]}...")
                ans = input("Skip this file? [Y/n] (or enter UUID to force-link to a transaction): ").strip().lower()
                if ans in ("", "y", "yes"):
                    # User confirms skip — just end this workflow
                    asyncio.run(receipt_processor.aupdate_state(cfg, {"human_approved": False}))
                    print("Skipped.\n")
                    continue
            else:
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

async def _get_mcp_session():
    """Create and return MCP server params for Actual Budget."""
    from mcp import StdioServerParameters
    return StdioServerParameters(
        command="node",
        args=["wrapper.js"],
        env=os.environ.copy()
    )

async def run_final_validation(fix=False):
    print(f"\n\n{'='*50}\nFINAL VALIDATION{' (with --fix)' if fix else ''}\n{'='*50}")
    # 1. Pull PB records (need tx_id and record id for deletion)
    pb_records = {}  # tx_id -> pb record id
    from tools.pocketbase_client import get_pb_client
    try:
        client = get_pb_client()
        records = client.collection("receipts").get_full_list(query_params={"fields": "id,tx_id"})
        for r in records:
            if hasattr(r, 'tx_id') and r.tx_id and r.tx_id != "unknown_id":
                pb_records[r.tx_id] = r.id
    except Exception as e:
        print(f"Failed to pull from PocketBase: {e}")
        return

    pb_tx_ids = set(pb_records.keys())

    # 2. Pull Actual Budget tx ids (transactions with #receipt in notes)
    actual_tx_ids = set()
    resolved_pb_tx_ids = set()
    pb_id_map = {} # old_tx_id -> resolved_parent_tx_id
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    
    server_params = await _get_mcp_session()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Resolve PB ids to parents if they are children
                    for tx_id in pb_tx_ids:
                        resolved_id = tx_id
                        try:
                            res = await session.call_tool("actual_query", {"args": [
                                "run", "--table", "transactions", "--filter", json.dumps({"id": tx_id}),
                                "--select", "id,is_child,parent_id", "--format", "json"
                            ]})
                            if res.content and len(res.content) > 0 and hasattr(res.content[0], 'text'):
                                query_data = json.loads(res.content[0].text)
                                if isinstance(query_data, list) and len(query_data) > 0:
                                    tx = query_data[0]
                                    if tx.get("is_child") and tx.get("parent_id"):
                                        resolved_id = tx.get("parent_id")
                        except Exception as e:
                            print(f"Failed to resolve PB tx {tx_id}: {e}")
                        
                        pb_id_map[tx_id] = resolved_id
                        resolved_pb_tx_ids.add(resolved_id)
                        
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
            break  # success
        except Exception as e:
            if "database is locked" in str(e).lower() or "sqlite" in str(e).lower():
                import asyncio
                wait = 2 ** attempt
                print(f"⏳ Actual Budget DB busy, retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            elif "TaskGroup" in str(e) or "JSONRPCMessage" in str(e) or "BrokenResourceError" in str(e):
                print("Ignored MCP shutdown error during query.")
                break
            else:
                print(f"Failed to pull from Actual Budget: {e}")
                return

    # 3. Find mismatches
    in_pb_not_actual = resolved_pb_tx_ids - actual_tx_ids
    in_actual_not_pb = actual_tx_ids - resolved_pb_tx_ids
    
    if not in_pb_not_actual and not in_actual_not_pb:
        print("✅ Validation passed. All receipts perfectly synced.")
        return

    if in_pb_not_actual:
        print(f"⚠️ {len(in_pb_not_actual)} receipts in PocketBase but tx is MISSING #receipt in Actual Budget:")
        for tx in in_pb_not_actual:
            print(f"  - {tx}")
    if in_actual_not_pb:
        print(f"⚠️ {len(in_actual_not_pb)} transactions in Actual Budget have #receipt tag but are MISSING in PocketBase:")
        for tx in in_actual_not_pb:
            print(f"  - {tx}")

    if not fix:
        return

    # ── Fix Phase ──
    print(f"\n{'─'*50}\nApplying fixes...\n{'─'*50}")

    # Fix PocketBase records that point to child transactions
    if pb_id_map:
        for old_tx_id, new_tx_id in pb_id_map.items():
            if old_tx_id != new_tx_id:
                pb_id = pb_records.get(old_tx_id)
                if pb_id:
                    try:
                        client.collection("receipts").update(pb_id, {"tx_id": new_tx_id})
                        print(f"  ✅ Updated PocketBase record {pb_id} to point to parent tx {new_tx_id} (was child {old_tx_id})")
                        # Update the in_pb_not_actual tracking so we can still delete if needed
                        if old_tx_id in pb_records:
                            pb_records[new_tx_id] = pb_records.pop(old_tx_id)
                    except Exception as e:
                        print(f"  ❌ Failed to update PB record {pb_id}: {e}")

    # Fix 1 & 3: For tx_ids in PB but not in Actual's #receipt list,
    # check if the tx actually exists in Actual. If yes → add #receipt. If no → delete PB record.
    if in_pb_not_actual:
        print(f"\n🔧 Checking {len(in_pb_not_actual)} PocketBase records against Actual Budget...")
        for attempt in range(max_retries):
            try:
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        from langchain_mcp_adapters.tools import load_mcp_tools
                        mcp_tools = await load_mcp_tools(session)
                        tx_tool = next((t for t in mcp_tools if t.name == "actual_transactions"), None)

                        for tx_id in in_pb_not_actual:
                            # Query Actual to see if the tx exists at all
                            try:
                                res = await session.call_tool("actual_query", {"args": [
                                    "run", "--table", "transactions",
                                    "--filter", json.dumps({"id": tx_id}),
                                    "--select", "id,notes,is_parent", "--format", "json"
                                ]})
                                tx_exists = False
                                existing_notes = ""
                                is_parent = False
                                if res.content and len(res.content) > 0 and hasattr(res.content[0], 'text'):
                                    query_data = json.loads(res.content[0].text)
                                    if isinstance(query_data, list) and len(query_data) > 0:
                                        tx_exists = True
                                        existing_notes = query_data[0].get("notes") or ""
                                        is_parent = query_data[0].get("is_parent")

                                if tx_exists:
                                    # Fix 3: tx exists but missing #receipt → add it
                                    if "#receipt" not in existing_notes:
                                        enriched_notes = ""
                                        if is_parent:
                                            try:
                                                c_res = await session.call_tool("actual_query", {"args": [
                                                    "run", "--table", "transactions", "--filter", json.dumps({"parent_id": tx_id}), 
                                                    "--select", "amount,category.name", "--format", "json"
                                                ]})
                                                if c_res.content and len(c_res.content) > 0:
                                                    children = json.loads(c_res.content[0].text)
                                                    if children:
                                                        enriched_notes += "Split breakdown:\n"
                                                        for child in children:
                                                            cat_name = child.get("category.name") or "Uncategorized"
                                                            amt = abs(child.get("amount", 0)) / 100
                                                            enriched_notes += f"- {cat_name}: ${amt:.2f}\n"
                                            except Exception as e:
                                                print(f"Failed to fetch child transactions: {e}")
                                                
                                        enriched_notes += "#receipt"
                                        
                                        if existing_notes.strip():
                                            new_notes = f"{existing_notes.strip()}\n\n{enriched_notes}"
                                        else:
                                            new_notes = enriched_notes
                                            
                                        if tx_tool:
                                            await tx_tool.ainvoke({"args": [
                                                "update", tx_id,
                                                "--data", json.dumps({"notes": new_notes})
                                            ]})
                                        print(f"  ✅ Added #receipt to tx {tx_id}")
                                    else:
                                        print(f"  ℹ️  tx {tx_id} already has #receipt (query race?)")
                                else:
                                    # Fix 2: tx not found in Actual → delete PB record
                                    pb_id = pb_records.get(tx_id)
                                    if pb_id:
                                        try:
                                            client.collection("receipts").delete(pb_id)
                                            print(f"  🗑️  Deleted PocketBase record {pb_id} (tx {tx_id} not in Actual)")
                                        except Exception as e:
                                            print(f"  ❌ Failed to delete PB record {pb_id}: {e}")
                            except Exception as e:
                                if "database is locked" in str(e).lower() or "sqlite" in str(e).lower():
                                    raise  # re-raise to trigger retry
                                print(f"  ❌ Error checking tx {tx_id}: {e}")
                break  # success
            except Exception as e:
                if "database is locked" in str(e).lower() or "sqlite" in str(e).lower():
                    import asyncio
                    wait = 2 ** attempt
                    print(f"⏳ Actual Budget DB busy, retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(wait)
                elif "TaskGroup" in str(e) or "JSONRPCMessage" in str(e) or "BrokenResourceError" in str(e):
                    print("Ignored MCP shutdown error (fixes were applied).")
                    break
                else:
                    print(f"❌ Failed during fix phase: {e}")
                    break

    # Fix 1: For tx_ids in Actual with #receipt but not in PB → remove #receipt
    if in_actual_not_pb:
        print(f"\n🔧 Removing #receipt from {len(in_actual_not_pb)} transactions not in PocketBase...")
        for attempt in range(max_retries):
            try:
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        from langchain_mcp_adapters.tools import load_mcp_tools
                        mcp_tools = await load_mcp_tools(session)
                        tx_tool = next((t for t in mcp_tools if t.name == "actual_transactions"), None)

                        for tx_id in in_actual_not_pb:
                            try:
                                # Fetch current notes
                                res = await session.call_tool("actual_query", {"args": [
                                    "run", "--table", "transactions",
                                    "--filter", json.dumps({"id": tx_id}),
                                    "--select", "id,notes", "--format", "json"
                                ]})
                                if res.content and len(res.content) > 0 and hasattr(res.content[0], 'text'):
                                    query_data = json.loads(res.content[0].text)
                                    if isinstance(query_data, list) and len(query_data) > 0:
                                        existing_notes = query_data[0].get("notes") or ""
                                        new_notes = existing_notes.replace("#receipt", "").strip()
                                        if tx_tool:
                                            await tx_tool.ainvoke({"args": [
                                                "update", tx_id,
                                                "--data", json.dumps({"notes": new_notes})
                                            ]})
                                        print(f"  ✅ Removed #receipt from tx {tx_id}")
                            except Exception as e:
                                if "database is locked" in str(e).lower() or "sqlite" in str(e).lower():
                                    raise
                                print(f"  ❌ Error fixing tx {tx_id}: {e}")
                break  # success
            except Exception as e:
                if "database is locked" in str(e).lower() or "sqlite" in str(e).lower():
                    import asyncio
                    wait = 2 ** attempt
                    print(f"⏳ Actual Budget DB busy, retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(wait)
                elif "TaskGroup" in str(e) or "JSONRPCMessage" in str(e) or "BrokenResourceError" in str(e):
                    print("Ignored MCP shutdown error (fixes were applied).")
                    break
                else:
                    print(f"❌ Failed during fix phase: {e}")
                    break

    print(f"\n{'─'*50}\nFix phase complete.")

def dump_cache_command(dump_path: str):
    import os, json, re, shutil
    from graph.cache import _get_client, _file_hash
    
    os.makedirs(dump_path, exist_ok=True)
    
    # clear existing files in dump_path
    for filename in os.listdir(dump_path):
        filepath = os.path.join(dump_path, filename)
        if os.path.isfile(filepath):
            os.unlink(filepath)
            
    client = _get_client()
    
    # Build hash to name mapping from local cache dir
    hash_to_name = {}
    local_cache_dir = os.path.join(os.getcwd(), "cache")
    if os.path.exists(local_cache_dir):
        for filename in os.listdir(local_cache_dir):
            file_path = os.path.join(local_cache_dir, filename)
            if os.path.isfile(file_path):
                try:
                    file_hash = _file_hash(file_path)
                    hash_to_name[file_hash] = filename
                except Exception:
                    pass
                
    # Also fetch from PocketBase as fallback
    try:
        from tools.pocketbase_client import get_pb_client
        pb_client = get_pb_client()
        page = 1
        per_page = 500
        while True:
            records = pb_client.collection("receipts").get_list(page, per_page, query_params={
                "fields": "source_hash,source_ref",
                "skipTotal": "true",
            })
            for r in records.items:
                h = getattr(r, "source_hash", None)
                ref = getattr(r, "source_ref", None)
                if h and ref and h not in hash_to_name:
                    hash_to_name[h] = ref.split('/')[-1]
            if len(records.items) < per_page:
                break
            page += 1
    except Exception as e:
        pass
        
    keys = client.keys("receipt:*:ocr")
    for key in keys:
        file_hash = key.split(":")[1]
        
        ocr_text = client.get(key)
        parsed_text = client.get(f"receipt:{file_hash}:parsed")
        
        original_name = hash_to_name.get(file_hash, f"unknown_{file_hash[:8]}")
        # Make safe filename, swap extension to .txt
        safe_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', original_name)
        safe_name = os.path.splitext(safe_name)[0] + '.txt'
            
        # prepend source as per user request (e.g. gdrive_)
        if not safe_name.startswith("gdrive_"):
            safe_name = f"gdrive_{safe_name}"
            
        out_filepath = os.path.join(dump_path, safe_name)
        
        with open(out_filepath, "w") as out:
            out.write("=== Raw OCR Text (Model: gemini-3.1-flash-lite) ===\n")
            out.write(ocr_text or "")
            out.write("\n\n=== Parsed JSON (Model: deepseek-v4-flash) ===\n")
            if parsed_text:
                try:
                    parsed_obj = json.loads(parsed_text)
                    out.write(json.dumps(parsed_obj, indent=2))
                except:
                    out.write(parsed_text)
            else:
                out.write("None")
                
    print(f"Dumped {len(keys)} cache entries to {dump_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receipt processing pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # `validate` subcommand
    validate_parser = subparsers.add_parser("validate", help="Run final validation between PocketBase and Actual Budget")
    validate_parser.add_argument("--fix", action="store_true", help="Attempt to fix mismatches automatically")

    # `cache` subcommand
    cache_parser = subparsers.add_parser("cache", help="Manage the Redis cache")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command")
    
    cache_dump_parser = cache_subparsers.add_parser("dump", help="Dump Redis nodecache to a directory")
    cache_dump_parser.add_argument("path", help="Directory path to dump the cache files")

    args = parser.parse_args()

    if args.command == "validate":
        import asyncio
        asyncio.run(run_final_validation(fix=args.fix))
    elif args.command == "cache":
        if args.cache_command == "dump":
            dump_cache_command(args.path)
        else:
            cache_parser.print_help()
    else:
        main()
