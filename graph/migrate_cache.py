"""Migrate existing filesystem cache (.ocr.txt / .parsed.json) into Redis."""

import glob
import json
import os
import sys

# Add project root to path so we can import graph.cache
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from graph.cache import set_ocr, set_parsed, _file_hash, _get_client

TMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")


def migrate():
    client = _get_client()
    client.ping()
    print(f"Connected to Redis. Migrating from {TMP_DIR} ...")

    ocr_files = glob.glob(os.path.join(TMP_DIR, "*.ocr.txt"))
    parsed_files = glob.glob(os.path.join(TMP_DIR, "*.parsed.json"))

    migrated_ocr = 0
    migrated_parsed = 0

    for ocr_path in ocr_files:
        # e.g. /path/to/foo.pdf.ocr.txt  →  /path/to/foo.pdf
        pdf_path = ocr_path.rsplit(".ocr.txt", 1)[0]
        if not os.path.isfile(pdf_path):
            print(f"  SKIP (PDF missing): {ocr_path}")
            continue
        with open(ocr_path, "r") as f:
            text = f.read()
        set_ocr(pdf_path, text)
        migrated_ocr += 1

    for parsed_path in parsed_files:
        pdf_path = parsed_path.rsplit(".parsed.json", 1)[0]
        if not os.path.isfile(pdf_path):
            print(f"  SKIP (PDF missing): {parsed_path}")
            continue
        with open(parsed_path, "r") as f:
            data = json.load(f)
        set_parsed(pdf_path, data)
        migrated_parsed += 1

    print(f"Done. Migrated {migrated_ocr} OCR entries and {migrated_parsed} parsed entries to Redis.")


if __name__ == "__main__":
    migrate()
