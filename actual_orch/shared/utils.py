import os
import json

def get_receipt_type(source: str) -> str:
    """
    Returns the generic receipt type for a given source.
    All file sources (file, dropbox, gdrive, etc.) are treated as 'file' based receipts,
    except for 'email' and 'paperless' which carry structured metadata.
    """
    if source == "email":
        return "email"
    if source == "paperless":
        return "paperless"
    return "file"

def get_config_path(filename: str) -> str:
    """
    Finds a config file in the project config directory or fallback locations.
    """
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config", filename)),
        os.path.abspath(os.path.join(os.getcwd(), "config", filename)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../config", filename)),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

def get_agent_config(node_name: str) -> dict:
    """
    Loads agents.json and returns the configuration for the specified node.
    Returns an empty dict if the config file or node doesn't exist.
    """
    config_path = get_config_path("agents.json")
    if not os.path.exists(config_path):
        return {}
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get(node_name, {})
    except Exception as e:
        print(f"  ⚠️ Failed to load agents config: {e}")
        return {}
