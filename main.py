import os
import sys
import json
import subprocess
import tempfile
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from graph.workflow import workflow

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
    print(f"file://{local_path}")
    
    async def run_flow():
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from graph.workflow import workflow
        import os
        
        os.makedirs("data", exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string("data/checkpoints.sqlite") as memory:
            receipt_processor = workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])
            
            # Step 1: Initial invocation
            result = await receipt_processor.ainvoke(state, config)
            
            # Step 2: Check if interrupted
            current_state = await receipt_processor.aget_state(config)
            if current_state.next and "human_approval" in current_state.next:
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

            # Dedup by file content hash (O(1) set lookup, no HTTP)
            from graph.cache import _file_hash
            file_hash = _file_hash(local_path)
            if file_hash in known_hashes:
                print(f"Skipping {gdrive_source_ref}: already processed in PocketBase (hash {file_hash[:12]}...).")
                continue

            res = process_receipt(local_path, gdrive_source_ref)
            if res and res.get("status") == "paused":
                paused_receipts.append((gdrive_source_ref, local_path, res["config"], res["state"]))
            print(f"--- Finished processing for {gdrive_source_ref} ---\n")
                
            # Optionally, move the file to a 'Processed' folder in gdrive
            # subprocess.check_call(["rclone", "move", gdrive_source_ref, f"{gdrive_path}/Processed/"])
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            print(f"--- Finished processing for {gdrive_source_ref} (Error) ---\n")

    if paused_receipts:
        print(f"\n{len(paused_receipts)} receipts require manual review.")
        print("Run 'python main.py review' to process them.")
    else:
        print("\nAll receipts processed successfully! No manual reviews needed.")
        
    try:
        import asyncio
        asyncio.run(run_final_validation())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting immediately to prevent background tasks.")
        os._exit(1)

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
    # 1. Pull PB records
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

    # 2. Fetch all transactions from Actual Budget
    actual_db = {}
    
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    
    server_params = await _get_mcp_session()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    print("Fetching accounts...")
                    res_accounts = await session.call_tool("actual_accounts", {"args": ["list"]})
                    if not res_accounts.content:
                        print("No accounts returned from Actual Budget.")
                        return
                    accounts = json.loads(res_accounts.content[0].text)
                    
                    print(f"Fetching all transactions across {len(accounts)} accounts...")
                    for acc in accounts:
                        res_list = await session.call_tool("actual_transactions", {"args": [
                            "list", "--account", acc["id"],
                            "--start", "1970-01-01", "--end", "2099-12-31",
                            "--format", "json"
                        ]})
                        if res_list.content and len(res_list.content) > 0:
                            txs = json.loads(res_list.content[0].text)
                            for tx in txs:
                                actual_db[tx.get("id")] = tx
                                # Also index subtransactions for easier lookups
                                for sub in tx.get("subtransactions", []):
                                    actual_db[sub.get("id")] = sub
            break
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

    # 3. Analyze data and resolve IDs
    resolved_pb_tx_ids = set()
    pb_id_map = {} # old_tx_id -> resolved_parent_tx_id
    
    for tx_id in pb_tx_ids:
        resolved_id = tx_id
        if tx_id in actual_db:
            tx = actual_db[tx_id]
            if tx.get("is_child") and tx.get("parent_id"):
                resolved_id = tx.get("parent_id")
        pb_id_map[tx_id] = resolved_id
        resolved_pb_tx_ids.add(resolved_id)
        
    actual_tx_ids = set()
    for tx_id, tx in actual_db.items():
        notes = tx.get("notes") or ""
        if "#receipt" in notes:
            if tx.get("is_child") and tx.get("parent_id"):
                actual_tx_ids.add(tx.get("parent_id"))
            else:
                actual_tx_ids.add(tx_id)
                
    # 4. Find mismatches
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
                            if tx_id in actual_db:
                                tx = actual_db[tx_id]
                                existing_notes = tx.get("notes") or ""
                                if "#receipt" not in existing_notes:
                                    enriched_notes = ""
                                    if tx.get("is_parent"):
                                        children = tx.get("subtransactions", [])
                                        if children:
                                            enriched_notes += "Split breakdown:\n"
                                            for child in children:
                                                cat_name = child.get("category.name") or "Uncategorized"
                                                amt = abs(child.get("amount", 0)) / 100
                                                enriched_notes += f"- {cat_name}: ${amt:.2f}\n"
                                                
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
                                # tx not found in Actual → delete PB record
                                pb_id = pb_records.get(tx_id)
                                if pb_id:
                                    try:
                                        client.collection("receipts").delete(pb_id)
                                        print(f"  🗑️  Deleted PocketBase record {pb_id} (tx {tx_id} not in Actual)")
                                    except Exception as e:
                                        print(f"  ❌ Failed to delete PB record {pb_id}: {e}")
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
                            if tx_id in actual_db:
                                existing_notes = actual_db[tx_id].get("notes") or ""
                                new_notes = existing_notes.replace("#receipt", "").strip()
                                if tx_tool:
                                    await tx_tool.ainvoke({"args": [
                                        "update", tx_id,
                                        "--data", json.dumps({"notes": new_notes})
                                    ]})
                                print(f"  ✅ Removed #receipt from tx {tx_id}")
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
            out.write("=== Metadata ===\n")
            out.write(f"ID: {file_hash}\n")
            out.write(f"Original File: {original_name}\n\n")
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

def prune_cache_command():
    from tools.pocketbase_client import fetch_all_source_hashes
    from graph.cache import _get_client
    import os
    
    known_hashes = fetch_all_source_hashes()
    client = _get_client()
    
    deleted_redis = 0
    for file_hash in known_hashes:
        keys = client.keys(f"receipt:{file_hash}:*")
        if keys:
            client.delete(*keys)
            deleted_redis += len(keys)
            
    print(f"Pruned {deleted_redis} Redis cache entries for already processed receipts.")
    
    local_cache_dir = os.path.join(os.getcwd(), "cache")
    deleted_local = 0
    if os.path.exists(local_cache_dir):
        from graph.cache import _file_hash
        for filename in os.listdir(local_cache_dir):
            file_path = os.path.join(local_cache_dir, filename)
            if os.path.isfile(file_path):
                try:
                    fhash = _file_hash(file_path)
                    if fhash in known_hashes:
                        os.unlink(file_path)
                        deleted_local += 1
                except Exception:
                    pass
    
    print(f"Pruned {deleted_local} local cache files for already processed receipts.")

def clear_cache_command():
    from graph.cache import _get_client
    import os, shutil
    
    client = _get_client()
    keys = client.keys("receipt:*")
    if keys:
        client.delete(*keys)
        print(f"Cleared {len(keys)} entries from Redis cache.")
    else:
        print("Redis cache is already empty.")
    
    local_cache_dir = os.path.join(os.getcwd(), "cache")
    if os.path.exists(local_cache_dir):
        shutil.rmtree(local_cache_dir)
        print(f"Cleared local file cache at {local_cache_dir}")
    else:
        print("Local cache is already empty.")

def purge_cache_command(ids: list[str]):
    from graph.cache import _get_client, _file_hash
    import os
    
    if not ids:
        print("No IDs provided to purge.")
        return
        
    client = _get_client()
    ids_set = set(ids)
    
    deleted_redis = 0
    for file_hash in ids_set:
        keys = client.keys(f"receipt:{file_hash}:*")
        if keys:
            client.delete(*keys)
            deleted_redis += len(keys)
            
    print(f"Purged {deleted_redis} Redis cache entries for {len(ids_set)} specified IDs.")
    
    local_cache_dir = os.path.join(os.getcwd(), "cache")
    deleted_local = 0
    if os.path.exists(local_cache_dir):
        for filename in os.listdir(local_cache_dir):
            file_path = os.path.join(local_cache_dir, filename)
            if os.path.isfile(file_path):
                try:
                    fhash = _file_hash(file_path)
                    if fhash in ids_set:
                        os.unlink(file_path)
                        deleted_local += 1
                except Exception:
                    pass
                    
    print(f"Purged {deleted_local} local cache files for specified IDs.")

def flag_source_command(filename: str, delete: bool = False):
    gdrive_path = os.environ.get("RECEIPT_GDRIVE_PATH")
    non_receipt_path = os.environ.get("NON_RECEIPT_GDRIVE_PATH")
    
    if not gdrive_path:
        print("RECEIPT_GDRIVE_PATH not set in .env")
        return
    if not non_receipt_path:
        print("NON_RECEIPT_GDRIVE_PATH not set in .env")
        return
        
    tmpdir = os.path.join(os.getcwd(), "cache")
    os.makedirs(tmpdir, exist_ok=True)
    
    local_path = os.path.join(tmpdir, filename)
    if not os.path.exists(local_path):
        try:
            local_path = download_file(gdrive_path, filename, tmpdir)
        except Exception as e:
            print(f"Failed to download file {filename} from {gdrive_path}: {e}")
            return
            
    from graph.cache import _file_hash
    try:
        file_hash = _file_hash(local_path)
    except Exception as e:
        print(f"Failed to compute hash for {local_path}: {e}")
        return
        
    from tools.pocketbase_client import delete_receipt_by_hash
    deleted = delete_receipt_by_hash(file_hash)
    if deleted:
        print(f"Deleted PocketBase record with hash {file_hash}")
    else:
        print(f"No PocketBase record found with hash {file_hash}")
        
    source_ref = f"{gdrive_path}/{filename}"
    
    if delete:
        print(f"Deleting {source_ref}...")
        try:
            subprocess.check_call(["rclone", "deletefile", source_ref])
            print("Delete completed.")
        except Exception as e:
            print(f"Failed to delete file using rclone: {e}")
    else:
        print(f"Moving {source_ref} to {non_receipt_path}...")
        try:
            subprocess.check_call(["rclone", "move", source_ref, non_receipt_path])
            print("Move completed.")
        except Exception as e:
            print(f"Failed to move file using rclone: {e}")

def flag_pb_command(pb_id: str):
    from tools.pocketbase_client import get_pb_client
    
    client = get_pb_client()
    try:
        record = client.collection("receipts").get_one(pb_id)
        source_hash = getattr(record, "source_hash", None)
        
        if source_hash:
            # Purge cache for this hash
            purge_cache_command([source_hash])
        else:
            print(f"No source_hash found for PocketBase record {pb_id}")
        
        # Delete from pocketbase
        client.collection("receipts").delete(pb_id)
        print(f"Deleted PocketBase record {pb_id}")
    except Exception as e:
        print(f"Failed to flag and delete PocketBase record {pb_id}: {e}")

async def areview_command():
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from graph.workflow import workflow
    import urllib.parse
    import re
    import os
    
    os.makedirs("data", exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string("data/checkpoints.sqlite") as memory:
        receipt_processor = workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])
        
        # Collect all unique thread IDs
        thread_ids = set()
        async for chk in memory.alist(None):
            thread_id = chk.config.get("configurable", {}).get("thread_id")
            if thread_id:
                thread_ids.add(thread_id)
                
        paused_states = []
        for thread_id in thread_ids:
            cfg = {"configurable": {"thread_id": thread_id}}
            snap = await receipt_processor.aget_state(cfg)
            if snap.next and "human_approval" in snap.next:
                paused_states.append(snap)
                
        if not paused_states:
            print("No receipts waiting for manual review.")
            return
            
        print(f"\n\n{'='*50}\nINTERACTIVE REVIEW PHASE ({len(paused_states)} receipts)\n{'='*50}")
        for snap in paused_states:
            cfg = snap.config
            st = snap
            ref = cfg["configurable"]["thread_id"]
            local_path = st.values.get("file_path", "")
            
            print(f"\nReviewing: {ref}")
            if local_path:
                encoded_path = urllib.parse.quote(local_path)
                print(f"File: 'file://{encoded_path}'")
            
            not_a_receipt = st.values.get("not_a_receipt", False)
            
            try:
                ans = ""
                if not_a_receipt:
                    print(f"🚫 Content was classified as NOT a receipt.")
                    print(f"   OCR preview: {(st.values.get('ocr_text') or '')[:200]}...")
                    ans = input("Action (s=skip, m=move, d=delete, UUID=link): [s] ").strip().lower()
                    if ans in ("", "s", "skip"):
                        ans = "s"
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
                        ans = input("Action (c=create, s=skip, m=move, d=delete, UUID=link): [c] ").strip().lower()
                        if ans in ("", "c", "create", "y", "yes"):
                            ans = "c"
                        elif ans in ("s", "skip", "n", "no"):
                            ans = "s"
                    else:
                        print(f"⚠️ Agent failed to match or propose a transaction.")
                        ans = input("Action (s=skip, m=move, d=delete, UUID=link): [s] ").strip().lower()
                        if ans in ("", "s", "skip", "n", "no"):
                            ans = "s"
                
                if ans in ("m", "d"):
                    filename = ref.split("/")[-1]
                    flag_source_command(filename, delete=(ans == "d"))
                    await receipt_processor.aupdate_state(cfg, {"human_approved": False})
                    await receipt_processor.ainvoke(None, cfg)
                    print("Done.\n")
                    continue
                elif ans == "s":
                    await receipt_processor.aupdate_state(cfg, {"human_approved": False})
                    await receipt_processor.ainvoke(None, cfg)
                    print("Skipped.\n")
                    continue
                elif re.match(r'^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$', ans, re.IGNORECASE):
                    await receipt_processor.aupdate_state(cfg, {"actual_tx_id": ans, "human_approved": False})
                else:
                    approved = (ans == "c")
                    await receipt_processor.aupdate_state(cfg, {"human_approved": approved})
                
                print(f"Resuming workflow for {ref}...")
                await receipt_processor.ainvoke(None, cfg)
                print("Done.\n")
            except KeyboardInterrupt:
                print("\nReview interrupted. State saved. Exiting immediately.")
                os._exit(1)

def review_command():
    import asyncio
    asyncio.run(areview_command())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receipt processing pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # `review` subcommand
    review_parser = subparsers.add_parser("review", help="Review paused receipts")

    # `validate` subcommand
    validate_parser = subparsers.add_parser("validate", help="Run final validation between PocketBase and Actual Budget")
    validate_parser.add_argument("--fix", action="store_true", help="Attempt to fix mismatches automatically")

    # `cache` subcommand
    cache_parser = subparsers.add_parser("cache", help="Manage the Redis cache")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command")
    
    cache_dump_parser = cache_subparsers.add_parser("dump", help="Dump Redis nodecache to a directory")
    cache_dump_parser.add_argument("path", help="Directory path to dump the cache files")

    cache_prune_parser = cache_subparsers.add_parser("prune", help="Remove cache for entries that exist in Pocketbase")
    cache_clear_parser = cache_subparsers.add_parser("clear", help="Remove all cache used by the application")
    
    cache_purge_parser = cache_subparsers.add_parser("purge", help="Remove cache for specific entry IDs")
    cache_purge_parser.add_argument("ids", nargs="+", help="List of cache entry IDs (hashes) to purge")

    # `flag` subcommand
    flag_parser = subparsers.add_parser("flag", help="Flag a file or pocketbase entry for deletion/cleanup")
    flag_subparsers = flag_parser.add_subparsers(dest="flag_command")
    
    flag_source_parser = flag_subparsers.add_parser("source", help="Flag a source file as not a receipt, delete from PB, move to non-receipts folder (or delete)")
    flag_source_parser.add_argument("filename", help="File name relative to RECEIPT_GDRIVE_PATH")
    flag_source_parser.add_argument("--delete", action="store_true", help="Delete the file instead of moving it to the non-receipts folder")

    flag_pb_parser = flag_subparsers.add_parser("pb", help="Flag a pocketbase entry, delete from PB, purge cache")
    flag_pb_parser.add_argument("pb_id", help="Pocketbase ID")

    args = parser.parse_args()

    try:
        if args.command == "validate":
            import asyncio
            asyncio.run(run_final_validation(fix=args.fix))
        elif args.command == "review":
            review_command()
        elif args.command == "cache":
            if args.cache_command == "dump":
                dump_cache_command(args.path)
            elif args.cache_command == "prune":
                prune_cache_command()
            elif args.cache_command == "clear":
                clear_cache_command()
            elif args.cache_command == "purge":
                purge_cache_command(args.ids)
            else:
                cache_parser.print_help()
        elif args.command == "flag":
            if args.flag_command == "source":
                flag_source_command(args.filename, args.delete)
            elif args.flag_command == "pb":
                flag_pb_command(args.pb_id)
            else:
                flag_parser.print_help()
        else:
            main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting immediately to prevent background tasks.")
        os._exit(1)
