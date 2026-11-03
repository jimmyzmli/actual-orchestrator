import os
import json
import tempfile
from actual_orch.clients.mcp import ActualMCPClient

# Kept for backward compatibility with external imports; internal logic no longer touches SQLite directly.
LOCAL_DB_PATH = None


class ReceiptRecord:
    """Lightweight compatibility wrapper mimicking PocketBase record objects."""
    def __init__(self, data: dict):
        self._data = dict(data) if isinstance(data, dict) else {}
        for k, v in self._data.items():
            setattr(self, k, v)

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item, value):
        self._data[item] = value
        setattr(self, item, value)

    def __contains__(self, item):
        return item in self._data

    def get(self, item, default=None):
        return self._data.get(item, default)

    def to_dict(self):
        return dict(self._data)

    def __repr__(self):
        return f"<ReceiptRecord id={self.get('id')} payee={self.get('payee')} amount={self.get('amount')}>"


class RecordList:
    """Wrapper mimicking PocketBase paginated record list."""
    def __init__(self, items: list, page: int = 1, per_page: int = 100, total_items: int = None):
        self.items = [ReceiptRecord(i) if not isinstance(i, ReceiptRecord) else i for i in items]
        self.page = page
        self.per_page = per_page
        self.total_items = total_items if total_items is not None else len(self.items)
        self.total_pages = 1 if len(self.items) <= per_page else (len(self.items) + per_page - 1) // per_page


class ReceiptsCollectionShim:
    """Encapsulates receipt operations via Actual MCP while exposing PocketBase collection API."""

    def get_list(self, page: int = 1, per_page: int = 100, query_params: dict = None):
        query_params = query_params or {}
        filter_expr = query_params.get("filter")
        fields = query_params.get("fields")
        limit = per_page
        offset = (page - 1) * per_page

        rows = ActualMCPClient.query_receipts_sync(
            filter=filter_expr,
            select=fields,
            limit=limit,
            offset=offset,
        )
        if not isinstance(rows, list):
            rows = []
        return RecordList(rows, page=page, per_page=per_page)

    def get_full_list(self, query_params: dict = None):
        query_params = query_params or {}
        filter_expr = query_params.get("filter")
        fields = query_params.get("fields")

        rows = ActualMCPClient.query_receipts_sync(
            filter=filter_expr,
            select=fields,
            limit=50000,
        )
        if not isinstance(rows, list):
            rows = []
        return [ReceiptRecord(r) for r in rows]

    def get_one(self, record_id: str, query_params: dict = None):
        row = ActualMCPClient.get_receipt_sync(id=record_id)
        if not row or not isinstance(row, dict) or not row.get("id"):
            raise ValueError(f"Receipt record not found: {record_id}")
        return ReceiptRecord(row)

    def _extract_image_paths(self, data: dict) -> list:
        image_paths = list(data.get("image_paths") or [])
        raw_image = data.get("image")
        if raw_image:
            files_to_save = []
            if hasattr(raw_image, "files"):
                files_to_save = raw_image.files
            elif isinstance(raw_image, (list, tuple)):
                files_to_save = raw_image

            for item in files_to_save:
                if isinstance(item, tuple) and len(item) >= 2:
                    name, content = item[0], item[1]
                    if isinstance(content, (bytes, bytearray)):
                        suffix = os.path.splitext(name)[1] or ".jpg"
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        tmp.write(content)
                        tmp.close()
                        image_paths.append(tmp.name)
        return image_paths

    def create(self, data: dict):
        payload = dict(data)
        image_paths = self._extract_image_paths(payload)
        if image_paths:
            payload["image_paths"] = image_paths
        payload.pop("image", None)

        res = ActualMCPClient.save_receipt_sync(payload)
        if not res or not isinstance(res, dict):
            raise RuntimeError(f"Failed to create receipt via Actual MCP: {res}")
        return ReceiptRecord(res)

    def update(self, record_id: str, data: dict):
        payload = dict(data)
        payload["id"] = record_id
        image_paths = self._extract_image_paths(payload)
        if image_paths:
            payload["image_paths"] = image_paths
        payload.pop("image", None)

        res = ActualMCPClient.save_receipt_sync(payload)
        if not res or not isinstance(res, dict):
            raise RuntimeError(f"Failed to update receipt {record_id} via Actual MCP: {res}")
        return ReceiptRecord(res)

    def delete(self, record_id: str):
        res = ActualMCPClient.delete_receipt_sync(id=record_id)
        return bool(res and isinstance(res, dict) and res.get("success"))


class PocketBaseShim:
    """Compatibility shim providing client.collection('receipts')."""
    def __init__(self):
        self._collections = {
            "receipts": ReceiptsCollectionShim()
        }

    def collection(self, name: str):
        if name not in self._collections:
            raise NotImplementedError(f"Actual MCP only encapsulates 'receipts' collection (requested: {name})")
        return self._collections[name]

    def auth_with_password(self, identity, password):
        return self


def get_pb_client():
    """Returns the Actual MCP PocketBase compatibility client."""
    return PocketBaseShim()


def fetch_all_field_values(field: str, filter_expr: str = None) -> set[str]:
    """
    Fetch all unique values for a specific field across receipts.
    Uses Actual MCP cached query, returning a set of strings.
    """
    rows = ActualMCPClient.query_receipts_sync(
        select=field,
        filter=filter_expr,
        limit=50000,
    )
    values = set()
    if isinstance(rows, list):
        for r in rows:
            if isinstance(r, dict):
                v = r.get(field)
                if v is not None and str(v).strip():
                    values.add(str(v).strip())
    return values


def fetch_all_receipts(filter_expr: str = None) -> list[dict]:
    """
    Fetch all receipts in a single sweep via Actual MCP.
    """
    if filter_expr:
        rows = ActualMCPClient.query_receipts_sync(filter=filter_expr, limit=50000)
    else:
        rows = ActualMCPClient.list_receipts_sync(limit=50000)

    if not isinstance(rows, list):
        return []

    receipts = []
    for r in rows:
        d = dict(r)
        d.pop("image", None)
        d.pop("expand", None)
        receipts.append(d)
    return receipts


def fetch_receipts_by_tx_ids(tx_ids: list) -> dict[str, list[dict]]:
    """
    Bulk lookup of receipts by Actual transaction UUIDs using Actual MCP.
    Returns dict mapping tx_id -> list of receipt dicts.
    """
    if not tx_ids:
        return {}

    rows = ActualMCPClient.list_receipts_sync(tx_ids=tx_ids, limit=len(tx_ids) * 10)
    if not isinstance(rows, list):
        return {}

    by_tx = {}
    for r in rows:
        if isinstance(r, dict):
            tid = r.get("tx_id")
            if tid:
                by_tx.setdefault(tid, []).append(dict(r))
    return by_tx
