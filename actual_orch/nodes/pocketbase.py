import os
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
            if os.path.exists(path):
                with open(path, "rb") as f:
                    uploads.append((f"page_{idx}.jpg", f.read()))
        if uploads:
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

def check_receipt_exists(source_hash, _cached_hashes=None, active_only=False):
    """
    Returns True if a receipt with the given source_hash already exists in PocketBase.
    If _cached_hashes (a set) is provided, does an O(1) in-memory lookup instead
    of hitting the API.
    """
    if _cached_hashes is not None:
        return source_hash in _cached_hashes

    client = get_pb_client()
    try:
        filter_str = f"source_hash='{source_hash}'"
        if active_only:
            filter_str += " && flag !~ 'ignore'"
        records = client.collection("receipts").get_list(1, 1, query_params={
            "filter": filter_str,
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

def should_skip_receipt(file_path: str, source_ref: str, source: str = "file", known_refs: set = None, known_hashes: set = None, override_classification: str = None, skip_cache_types: list = None, document_date: str = None) -> tuple[bool, str]:
    """
    Consolidated check for whether a receipt should be skipped before processing.
    Checks PocketBase (source_ref, file_hash), Classification Cache, and Parsed Cache.
    Returns (should_skip, reason_string).
    """
    from actual_orch.shared.utils import get_receipt_type
    receipt_type = get_receipt_type(source)
    
    # 1. Check PocketBase by source_ref
    clean_ref = source_ref
    if receipt_type != "email" and clean_ref:
        clean_ref = clean_ref.split("/")[-1]
        
    if clean_ref:
        if known_refs is not None and clean_ref in known_refs:
            return True, f"already processed in PocketBase (ref {clean_ref})"
        elif known_refs is None:
            client = get_pb_client()
            try:
                records = client.collection("receipts").get_list(1, 1, query_params={
                    "filter": f"source_ref='{clean_ref}'",
                    "fields": "id,flag",
                    "skipTotal": "true",
                })
                if len(records.items) > 0:
                    flag = getattr(records.items[0], "flag", None)
                    if flag and ("ignore" in str(flag).lower() or (isinstance(flag, list) and any("ignore" in str(f).lower() for f in flag))):
                        return True, "flagged as ignore in PocketBase"
                    return True, f"already processed in PocketBase (ref {clean_ref})"
            except Exception as e:
                pass

    # 2. Check PocketBase by source_hash
    if file_path:
        file_hash = _file_hash(file_path)
        if known_hashes is not None and file_hash in known_hashes:
            return True, f"already processed in PocketBase (hash {file_hash[:12]}...)"
        elif known_hashes is None:
            client = get_pb_client()
            try:
                records = client.collection("receipts").get_list(1, 1, query_params={
                    "filter": f"source_hash='{file_hash}'",
                    "fields": "id,flag",
                    "skipTotal": "true",
                })
                if len(records.items) > 0:
                    flag = getattr(records.items[0], "flag", None)
                    if flag and ("ignore" in str(flag).lower() or (isinstance(flag, list) and any("ignore" in str(f).lower() for f in flag))):
                        return True, "flagged as ignore in PocketBase"
                    return True, f"already processed in PocketBase (hash {file_hash[:12]}...)"
            except Exception:
                pass

    # 3. Check Redis Caches (Classification and Parsed data)
    if file_path:
        from actual_orch.clients.cache import get_classify, get_parsed
        import datetime
        import json
        
        prefix = f"receipt_{source}_" if source else "receipt_"
        skip_cache_types = skip_cache_types or []
        
        # Check classification
        if "classify" not in skip_cache_types:
            check_class = override_classification if override_classification else get_classify(file_path, prefix=prefix)
            config_path_class = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/classification.json"))
            skip_classes = ["NOT_RECEIPT", "SHIPPING_CONFIRMATION", "INVOICE"]
            if os.path.exists(config_path_class):
                try:
                    with open(config_path_class, "r") as f:
                        class_data = json.load(f)
                        skip_classes = class_data.get("skip_classifications", skip_classes)
                except Exception:
                    pass
                    
            if check_class and any(skip_cls in check_class for skip_cls in skip_classes):
                return True, f"classified as {check_class.split('|')[0]}"
            
        # Check parsed data (0 amount or future date)
        if "parsed" not in skip_cache_types:
            cached_parsed = get_parsed(file_path, prefix=prefix)
            if not cached_parsed and prefix != "receipt_":
                cached_parsed = get_parsed(file_path, prefix="receipt_")
                
            if cached_parsed:
                cached_class = override_classification if override_classification else get_classify(file_path, prefix=prefix)
                is_zero = cached_parsed.get("total") is None or cached_parsed.get("total") == 0.0
                if is_zero:
                    class_str = cached_class if cached_class else "UNKNOWN"
                    return True, f"Total amount is 0. (Classified as: {class_str.split('|')[0]})"
                    
                if cached_parsed.get("date"):
                    try:
                        parsed_date = datetime.datetime.strptime(cached_parsed["date"], "%Y-%m-%d").date()
                        if datetime.date.today() < parsed_date:
                            # If a reference document date is known and not in the future, don't skip;
                            # it will be corrected by parse_receipt during processing.
                            if document_date:
                                ref_date = datetime.datetime.strptime(document_date, "%Y-%m-%d").date()
                                if ref_date <= datetime.date.today():
                                    pass
                                else:
                                    class_str = cached_class if cached_class else "UNKNOWN"
                                    return True, f"classified as {class_str.split('|')[0]} (future date)"
                            else:
                                class_str = cached_class if cached_class else "UNKNOWN"
                                return True, f"classified as {class_str.split('|')[0]} (future date)"
                    except Exception:
                        pass

    return False, ""

def skip_if_processed(state: dict):
    """Check PocketBase and caches to skip already-processed files."""
    if state.get("force_rerun"):
        return {}

    source = state.get("source", "file")
    source_ref = state.get("source_ref", "")
    file_path = state.get("file_path", "")
    override_classification = state.get("override_classification")
    skip_cache_types = state.get("flag_skip_cache_types", [])
    document_date = state.get("document_date")
    
    skip, reason = should_skip_receipt(
        file_path,
        source_ref,
        source,
        override_classification=override_classification,
        skip_cache_types=skip_cache_types,
        document_date=document_date
    )
    if skip:
        print(f"⏭️  Skipped {source_ref or file_path}: {reason}")
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
        print("[save_pocketbase] " + "Saving to PocketBase...")

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
        # Ensure image paths exist on disk; regenerate if missing
        valid_images = [p for p in image_paths if os.path.exists(p)]
        if not valid_images and file_path and os.path.exists(file_path):
            is_pdf = file_path.lower().endswith(".pdf")
            if not is_pdf:
                try:
                    with open(file_path, "rb") as f:
                        is_pdf = f.read(5).startswith(b"%PDF")
                except Exception:
                    pass
            if is_pdf:
                try:
                    from pdf2image import convert_from_path
                    gen_dir = os.path.join(os.getcwd(), "cache", "generated_image")
                    os.makedirs(gen_dir, exist_ok=True)
                    images = convert_from_path(file_path)
                    for idx, img in enumerate(images):
                        safe_name = os.path.basename(file_path) + f"_page_{idx}.jpg"
                        img_path = os.path.join(gen_dir, safe_name)
                        if not os.path.exists(img_path):
                            img.save(img_path, format="JPEG")
                        valid_images.append(img_path)
                except Exception as e:
                    print(f"[save_pocketbase][warning] Failed to generate page images: {e}")
            elif file_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".heic")):
                valid_images = [file_path]
        image_paths = valid_images

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
            for path in image_paths:
                if "cache/generated_image" in path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"[save_pocketbase][error] Warning: Failed to clean up {path}: {e}")
                        
        return {"status": "success"}
    except Exception as e:
        print(f"[save_pocketbase][error] ❌ PocketBase save failed: {e}")
        return {"status": "error", "error": f"PocketBase save failed: {str(e)}"}
