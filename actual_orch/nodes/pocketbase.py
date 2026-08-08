import json
from pocketbase.models.file_upload import FileUpload
from actual_orch.clients.pocketbase import get_pb_client
from actual_orch.clients.cache import _file_hash

def save_receipt(transaction_id, lines, tx_date, source_ref, amount, payee, source_hash, image_paths=None, tx_id_transfer=None, parsed=None, source="file"):
    """
    transaction_id: The original 36 char UUID from actual budget
    lines: list of extracted items
    tx_date: ISO date string
    source_ref: file path/filename
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
        "source": source,
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
        # Try to find existing record by source_hash or source_ref
        existing = None
        filter_expr = []
        if source_hash:
            filter_expr.append(f"source_hash='{source_hash}'")
        if source_ref:
            filter_expr.append(f"source_ref='{source_ref}'")
            
        if filter_expr:
            try:
                filter_str = " || ".join(filter_expr)
                records = client.collection("receipts").get_list(1, 1, query_params={
                    "filter": filter_str,
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
    """Check PocketBase source_hash or source_ref to skip already-processed files."""
    source = state.get("source", "file")
    
    # 1. Check source_ref
    source_ref = state.get("source_ref", "")
    from actual_orch.shared.utils import get_receipt_type
    receipt_type = get_receipt_type(source)
    if receipt_type != "email" and source_ref:
        source_ref = source_ref.split("/")[-1]
        
    if source_ref:
        client = get_pb_client()
        try:
            records = client.collection("receipts").get_list(1, 1, query_params={
                "filter": f"source_ref='{source_ref}'",
                "fields": "id",
                "skipTotal": "true",
            })
            if len(records.items) > 0:
                print(f"⏭️  Already processed in PocketBase (ref {source_ref})")
                return {"status": "already_processed"}
        except Exception:
            pass

    # 2. Check source_hash
    file_path = state.get("file_path", "")
    if file_path:
        file_hash = _file_hash(file_path)
        if check_receipt_exists(file_hash):
            print(f"⏭️  Already processed in PocketBase (hash {file_hash[:12]}...): {file_path}")
            return {"status": "already_processed"}
            
    return {}

def update_replaced_tx_id(replaced_tx_id, new_tx_id):
    """Update all receipts in PocketBase that referenced the old transaction ID to the new one"""
    client = get_pb_client()
    try:
        records = client.collection("receipts").get_full_list(query_params={
            "filter": f"tx_id='{replaced_tx_id}'",
        })
        for r in records:
            client.collection("receipts").update(r.id, {"tx_id": new_tx_id})
    except Exception as e:
        print(f"Warning: Failed to update replaced_tx_id {replaced_tx_id} in PocketBase: {e}")

def save_pocketbase_node(state: dict):
    """Save the receipt and items to PocketBase"""
    try:
        print("Saving to PocketBase...")

        parsed = state.get("parsed_data")
        tx_id = state.get("actual_tx_id")
        replaced_tx_id = state.get("replaced_tx_id")
        
        if replaced_tx_id and tx_id and replaced_tx_id != tx_id:
            update_replaced_tx_id(replaced_tx_id, tx_id)
        
        if not parsed or not tx_id or tx_id == "unknown_id":
            return {"status": "error", "error": "Missing parsed data or valid tx_id (was unknown_id)"}
            
        date = parsed["date"]
        amount = round(parsed["total"], 2) if parsed.get("total") is not None else 0.0
        payee = parsed["payee"]
        lines = parsed.get("items", [])
        
        source = state.get("source", "file")
        source_ref = state.get("source_ref", "")
        from actual_orch.shared.utils import get_receipt_type
        receipt_type = get_receipt_type(source)
        if receipt_type != "email" and source_ref:
            source_ref = source_ref.split("/")[-1]
            
        image_paths = state.get("image_paths", [])
        file_path = state.get("file_path", "")
        if source == "email":
            source_hash = ""
        else:
            source_hash = _file_hash(file_path) if file_path else ""
        tx_id_transfer = state.get("actual_tx_id_transfer")
        
        parsed_to_save = dict(parsed)
        parsed_to_save["schema_version"] = "1.0.0"
        
        res = save_receipt(tx_id, lines, date, source_ref, amount, payee, source_hash, image_paths, tx_id_transfer, parsed_to_save, source=source)
        
        # Clean up generated images
        if image_paths:
            import os
            for path in image_paths:
                if "cache/generated_image" in path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"Warning: Failed to clean up {path}: {e}")
                        
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "error": f"PocketBase save failed: {str(e)}"}
