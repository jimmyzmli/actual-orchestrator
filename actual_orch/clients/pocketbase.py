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
