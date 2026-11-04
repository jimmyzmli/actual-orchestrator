import os
import json
from actual_orch.clients.mcp import ActualMCPClient
from actual_orch.clients.cache import _file_hash


def save_receipt(transaction_id, lines, tx_date, source_ref, amount, payee, source_hash, image_paths=None, tx_id_transfer=None, parsed=None, source="file"):
    """
    Save or update a receipt in PocketBase and Actual MCP cache, linking it to an Actual Budget transaction.
    """
    payload = {
        "tx_id": transaction_id,
        "lines": lines,
        "date": tx_date,
        "amount": amount,
        "payee": payee,
        "source": source,
        "source_ref": source_ref,
        "source_hash": source_hash,
    }

    if tx_id_transfer:
        payload["tx_id_transfer"] = tx_id_transfer

    if parsed:
        payload["parsed"] = parsed

    if image_paths:
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        payload["image_paths"] = valid_paths
        
        # Also encode as base64 images so saving receipts works seamlessly from remote clients
        import base64
        import mimetypes
        encoded_images = []
        for p in valid_paths:
            try:
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                mime, _ = mimetypes.guess_type(p)
                encoded_images.append({
                    "filename": os.path.basename(p),
                    "mime_type": mime or "image/jpeg",
                    "data_base64": b64
                })
            except Exception as img_err:
                print(f"[save_receipt][warn] Failed to encode {p} for remote transfer: {img_err}")
        if encoded_images:
            payload["images"] = encoded_images

    try:
        res = ActualMCPClient.save_receipt_sync(payload)
        return res
    except Exception as e:
        raise Exception(f"Failed to save receipt via Actual MCP: {e}")


def check_receipt_exists(source_hash, _cached_hashes=None, active_only=False):
    """
    Returns True if a receipt with the given source_hash already exists in PocketBase.
    If _cached_hashes (a set) is provided, does an O(1) in-memory lookup instead.
    """
    if _cached_hashes is not None:
        return source_hash in _cached_hashes

    try:
        return ActualMCPClient.check_receipt_exists_sync(source_hash=source_hash, active_only=active_only)
    except Exception:
        return False


def delete_receipt_by_hash(source_hash):
    """
    Looks up a receipt by source_hash and deletes it if it exists.
    Returns True if a record was deleted, False otherwise.
    """
    try:
        res = ActualMCPClient.delete_receipt_sync(source_hash=source_hash)
        return bool(res and isinstance(res, dict) and res.get("success"))
    except Exception as e:
        print(f"Error deleting receipt from Actual MCP: {e}")
        return False


def should_skip_receipt(file_path: str, source_ref: str, source: str = "file", known_refs: set = None, known_hashes: set = None, override_classification: str = None, skip_cache_types: list = None, document_date: str = None) -> tuple[bool, str]:
    """
    Consolidated check for whether a receipt should be skipped before processing.
    Checks Actual MCP (source_ref, file_hash), Classification Cache, and Parsed Cache.
    Returns (should_skip, reason_string).
    """
    from actual_orch.shared.utils import get_receipt_type
    receipt_type = get_receipt_type(source)

    # 1. Check PocketBase / Actual MCP by source_ref
    clean_ref = source_ref
    if receipt_type != "email" and clean_ref:
        clean_ref = clean_ref.split("/")[-1]

    if clean_ref:
        if known_refs is not None and clean_ref in known_refs:
            return True, f"already processed in PocketBase (ref {clean_ref})"
        elif known_refs is None:
            try:
                rec = ActualMCPClient.get_receipt_sync(source_ref=clean_ref)
                if rec and isinstance(rec, dict) and rec.get("id"):
                    flag = rec.get("flag", "")
                    if flag and "ignore" in str(flag).lower():
                        return True, "flagged as ignore in PocketBase"
                    return True, f"already processed in PocketBase (ref {clean_ref})"
            except Exception:
                pass

    # 2. Check PocketBase / Actual MCP by source_hash
    if file_path:
        file_hash = _file_hash(file_path)
        if known_hashes is not None and file_hash in known_hashes:
            return True, f"already processed in PocketBase (hash {file_hash[:12]}...)"
        elif known_hashes is None:
            try:
                rec = ActualMCPClient.get_receipt_sync(source_hash=file_hash)
                if rec and isinstance(rec, dict) and rec.get("id"):
                    flag = rec.get("flag", "")
                    if flag and "ignore" in str(flag).lower():
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
                    with open(config_path_class, "r", encoding="utf-8") as f:
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
    """Check Actual MCP and caches to skip already-processed files."""
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
    """Update all receipts in Actual MCP that referenced the old transaction ID to the new one"""
    try:
        records = ActualMCPClient.list_receipts_sync(tx_id=replaced_tx_id)
        if isinstance(records, list):
            for r in records:
                if isinstance(r, dict) and r.get("id"):
                    ActualMCPClient.save_receipt_sync({"id": r["id"], "tx_id": new_tx_id})
    except Exception as e:
        print(f"Warning: Failed to update replaced_tx_id {replaced_tx_id} via Actual MCP: {e}")


def save_pocketbase_node(state: dict):
    """Save the receipt and items to Actual MCP / PocketBase"""
    try:
        print("[save_pocketbase] " + "Saving receipt...")

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

        save_receipt(tx_id, lines, date, source_ref, amount, payee, source_hash, image_paths, tx_id_transfer, parsed_to_save, source=source)

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
        print(f"[save_pocketbase][error] ❌ Receipt save failed: {e}")
        return {"status": "error", "error": f"Receipt save failed: {str(e)}"}
