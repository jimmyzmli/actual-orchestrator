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

def save_receipt(transaction_id, lines, tx_date, source_ref, amount, payee, image_paths=None):
    """
    transaction_id: The original 36 char UUID from actual budget
    lines: list of extracted items
    tx_date: ISO date string
    source_ref: gdrive path/filename
    amount: total amount
    payee: store name
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
        "source_ref": source_ref
    }
    
    if image_paths:
        uploads = []
        for idx, path in enumerate(image_paths):
            with open(path, "rb") as f:
                uploads.append((f"page_{idx}.jpg", f.read()))
        data["image"] = FileUpload(*uploads)
    
    try:
        # Check if exists by source_ref
        records = client.collection("receipts").get_list(1, 1, query_params={"filter": f"source_ref='{source_ref}'"})
        if records.items:
            # If exists, update
            pb_id = records.items[0].id
            return client.collection("receipts").update(pb_id, data)
        else:
            # If not exists, create
            return client.collection("receipts").create(data)
    except Exception as e:
        raise Exception(f"Failed to upsert to pocketbase: {e}")

def check_receipt_exists(source_ref):
    """
    Returns True if a receipt with the given source_ref already exists in PocketBase.
    """
    client = get_pb_client()
    try:
        records = client.collection("receipts").get_list(1, 1, query_params={"filter": f"source_ref='{source_ref}'"})
        return len(records.items) > 0
    except Exception:
        return False
