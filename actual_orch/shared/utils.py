def get_receipt_type(source: str) -> str:
    """
    Returns the generic receipt type for a given source.
    All file sources (file, dropbox, gdrive, etc.) are treated as 'file' based receipts,
    except for 'email' which is treated as an email receipt.
    """
    if source == "email":
        return "email"
    return "file"
