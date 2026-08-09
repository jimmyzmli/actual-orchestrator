import os
import json

def get_receipt_type(source: str) -> str:
    """
    Returns the generic receipt type for a given source.
    All file sources (file, dropbox, gdrive, etc.) are treated as 'file' based receipts,
    except for 'email' which is treated as an email receipt.
    """
    if source == "email":
        return "email"
    return "file"

def get_agent_config(node_name: str) -> dict:
    """
    Loads agents.json and returns the configuration for the specified node.
    Returns an empty dict if the config file or node doesn't exist.
    """
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../config/agents.json"))
    if not os.path.exists(config_path):
        return {}
        
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get(node_name, {})
    except Exception as e:
        print(f"  ⚠️ Failed to load agents config: {e}")
        return {}
