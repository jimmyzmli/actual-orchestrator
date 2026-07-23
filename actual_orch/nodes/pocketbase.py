import json
from pocketbase.models.file_upload import FileUpload
from actual_orch.clients.pocketbase import get_pb_client
from actual_orch.clients.cache import _file_hash

def save_receipt(transaction_id, lines, tx_date, source_ref, amount, payee, source_hash, image_paths=None, tx_id_transfer=None, parsed=None):
    """
    transaction_id: The original 36 char UUID from actual budget
    lines: list of extracted items
    tx_date: ISO date string
    source_ref: gdrive path/filename
    amount: total amount
    payee: store name
    source_hash: SHA-256 hash of the source file for dedup
    image_paths: optional list of local paths to jpeg images
    tx_id_transfer: The 36 char UUID of the other side of a transfer, if applicable
    parsed: the full parsed json dict
    """
    client = get_pb_client()
    
    data = {
        "tx_id": transaction_id,
        "lines": json.dumps(lines),
        "date": tx_date + " 00:00:00.000Z", # Pocketbase date format is 'YYYY-MM-DD HH:MM:SS.ZZZ'
        "amount": amount,
        "payee": payee,
        "source": "gdrive",
        "source_ref": source_ref,
        "source_hash": source_hash
    }
    
    if tx_id_transfer:
        data["tx_id_transfer"] = tx_id_transfer
        
    if parsed:
        data["parsed"] = json.dumps(parsed)
    
    if image_paths:
        uploads = []
        for idx, path in enumerate(image_paths):
            with open(path, "rb") as f:
                uploads.append((f"page_{idx}.jpg", f.read()))
        data["image"] = FileUpload(*uploads)
    
    try:
        # Try to find existing record by source_hash
        existing = None
        if source_hash:
            try:
                records = client.collection("receipts").get_list(1, 1, query_params={
                    "filter": f"source_hash~'{source_hash}'",
                    "fields": "id,source_hash,source_ref",
                    "skipTotal": "true",
                })
                if records.items:
                    existing = records.items[0]
            except Exception:
                pass

        if existing:
            return client.collection("receipts").update(existing.id, data)
        else:
            return client.collection("receipts").create(data)
    except Exception as e:
        raise Exception(f"Failed to upsert to pocketbase: {e}")

def check_receipt_exists(source_hash, _cached_hashes=None):
    """
    Returns True if a receipt with the given source_hash already exists in PocketBase.
    If _cached_hashes (a set) is provided, does an O(1) in-memory lookup instead
    of hitting the API.
    """
    if _cached_hashes is not None:
        return source_hash in _cached_hashes

    client = get_pb_client()
    try:
        records = client.collection("receipts").get_list(1, 1, query_params={
            "filter": f"source_hash='{source_hash}'",
            "fields": "id",
            "skipTotal": "true",
        })
        return len(records.items) > 0
    except Exception:
        return False

def delete_receipt_by_hash(source_hash):
    """
    Looks up a receipt by source_hash and deletes it if it exists.
    Returns True if a record was deleted, False otherwise.
    """
    client = get_pb_client()
    try:
        records = client.collection("receipts").get_list(1, 1, query_params={
            "filter": f"source_hash='{source_hash}'",
            "fields": "id",
            "skipTotal": "true",
        })
        if records.items:
            client.collection("receipts").delete(records.items[0].id)
            return True
    except Exception as e:
        print(f"Error deleting from pocketbase: {e}")
    return False

def skip_if_processed(state: dict):
    """Check PocketBase source_hash to skip already-processed files."""
    file_path = state.get("file_path", "")
    if file_path:
        file_hash = _file_hash(file_path)
        if check_receipt_exists(file_hash):
            print(f"⏭️  Already processed in PocketBase (hash {file_hash[:12]}...): {file_path}")
            return {"status": "already_processed"}
    return {}

def save_pocketbase_node(state: dict):
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
        if source_ref:
            source_ref = source_ref.split("/")[-1]
        image_paths = state.get("image_paths", [])
        file_path = state.get("file_path", "")
        source_hash = _file_hash(file_path) if file_path else ""
        tx_id_transfer = state.get("actual_tx_id_transfer")
        
        parsed_to_save = dict(parsed)
        parsed_to_save["schema_version"] = "1.0.0"
        
        res = save_receipt(tx_id, lines, date, source_ref, amount, payee, source_hash, image_paths, tx_id_transfer, parsed_to_save)
        
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "error": f"PocketBase save failed: {str(e)}"}
