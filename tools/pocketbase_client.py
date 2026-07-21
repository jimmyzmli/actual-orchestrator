import os
from pocketbase import PocketBase

def get_pb_client():
    pb_url = os.environ.get("PB_UPLOAD_BASE_URL")
    pb_user = os.environ.get("PB_USER")
    pb_pass = os.environ.get("PB_PASSWORD")
    
    if not pb_url or not pb_user or not pb_pass:
        raise ValueError("PocketBase credentials not found in environment")
        
    client = PocketBase(pb_url)
    client.collection("users").auth_with_password(pb_user, pb_pass)
    return client

from pocketbase.models.file_upload import FileUpload

def save_receipt(transaction_id, lines, tx_date, source_ref, amount, payee, source_hash, image_paths=None):
    """
    transaction_id: The original 36 char UUID from actual budget
    lines: list of extracted items
    tx_date: ISO date string
    source_ref: gdrive path/filename
    amount: total amount
    payee: store name
    source_hash: SHA-256 hash of the source file for dedup
    image_paths: optional list of local paths to jpeg images
    """
    client = get_pb_client()
    
    import json
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

def fetch_all_source_hashes():
    """
    Fetch all source_hash values from PocketBase in a single paginated sweep.
    Uses `fields=source_hash` and `skipTotal=true` to minimise payload and
    skip the expensive COUNT(*) query on the server.
    Returns a set of hash strings.
    """
    client = get_pb_client()
    hashes = set()
    page = 1
    per_page = 500
    while True:
        try:
            records = client.collection("receipts").get_list(page, per_page, query_params={
                "fields": "source_hash",
                "skipTotal": "true",
            })
            for r in records.items:
                h = getattr(r, "source_hash", None)
                if h:
                    hashes.add(h)
            # If fewer items than perPage, we've reached the last page
            if len(records.items) < per_page:
                break
            page += 1
        except Exception as e:
            print(f"⚠️ Error fetching source hashes page {page}: {e}")
            break
    return hashes


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
