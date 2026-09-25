import os
from dotenv import load_dotenv

# Automatically load .env from project root or current working directory if available
_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
if os.path.exists(_env_path):
    load_dotenv(_env_path)
else:
    load_dotenv()
