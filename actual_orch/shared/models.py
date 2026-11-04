import os
import json
from typing import Any, Dict, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from actual_orch.shared.utils import get_config_path

_MODELS_CACHE: Optional[Dict[str, Any]] = None

def load_models_config() -> Dict[str, Any]:
    """
    Loads models.json and returns the full configuration dictionary.
    """
    global _MODELS_CACHE
    config_path = get_config_path("models.json")
    if not os.path.exists(config_path):
        return {"models": {}, "default_ollama_base_url": "http://localhost:11434"}
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _MODELS_CACHE = json.load(f)
        return _MODELS_CACHE or {}
    except Exception as e:
        print(f"  ⚠️ Failed to load models config: {e}")
        return {"models": {}, "default_ollama_base_url": "http://localhost:11434"}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Retrieves configuration for a model name.
    Supports config-defined models and automatic defaults for known patterns.
    """
    config = load_models_config()
    models_dict = config.get("models", {})
    
    ollama_host = os.environ.get("OLLAMA_BASE_URL", config.get("default_ollama_base_url", "http://localhost:11434")).rstrip("/")
    if not ollama_host.endswith("/v1"):
        ollama_base_url = f"{ollama_host}/v1"
    else:
        ollama_base_url = ollama_host

    if model_name in models_dict:
        model_cfg = dict(models_dict[model_name])
        # If it's an ollama provider, respect OLLAMA_BASE_URL override if present
        if model_cfg.get("provider") == "ollama" and "OLLAMA_BASE_URL" in os.environ:
            model_cfg["base_url"] = ollama_base_url
        return model_cfg
        
    # Auto-fallback resolution if not explicitly in models.json
    if model_name.startswith("gemini"):
        return {
            "provider": "google",
            "model": model_name,
            "api_key_env": "GEMINI_API_KEY",
            "temperature": 0.0
        }
    elif ":" in model_name or model_name.startswith("ollama/"):
        real_model = model_name.split("ollama/")[-1] if model_name.startswith("ollama/") else model_name
        return {
            "provider": "ollama",
            "model": real_model,
            "base_url": ollama_base_url,
            "api_key": "ollama",
            "temperature": 0.0
        }
    elif model_name.startswith("deepseek"):
        return {
            "provider": "openai",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "api_key_env": "DEEPSEEK_API_KEY",
            "peak_hours": "deepseek"
        }
    else:
        # Default fallback
        return {
            "provider": "openai",
            "model": model_name,
            "base_url": "https://api.deepseek.com/v1",
            "api_key_env": "DEEPSEEK_API_KEY",
            "peak_hours": "deepseek"
        }

def get_chat_model(model_name: str, check_peak_hours: bool = True, **kwargs) -> BaseChatModel:
    """
    Instantiates a LangChain ChatModel based on models.json configuration.
    """
    model_cfg = get_model_config(model_name)
    
    # Check peak hours if model specifies a schedule
    peak_schedule = model_cfg.get("peak_hours")
    if check_peak_hours and peak_schedule:
        from actual_orch.shared.peak_hours import pause_if_peak_hours
        pause_if_peak_hours(peak_schedule)
        
    provider = model_cfg.get("provider", "openai")
    
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key_env = model_cfg.get("api_key_env", "GEMINI_API_KEY")
        api_key = os.environ.get(api_key_env, "")
        
        params: Dict[str, Any] = {
            "model": model_cfg.get("model", model_name),
            "api_key": api_key,
        }
        if "temperature" in model_cfg:
            params["temperature"] = model_cfg["temperature"]
        params.update(kwargs)
        return ChatGoogleGenerativeAI(**params)
        
    elif provider in ("ollama", "openai"):
        from langchain_openai import ChatOpenAI
        
        api_key = model_cfg.get("api_key")
        if not api_key and "api_key_env" in model_cfg:
            api_key = os.environ.get(model_cfg["api_key_env"], "")
        if not api_key:
            api_key = "ollama" if provider == "ollama" else os.environ.get("OPENAI_API_KEY", "")
            
        base_url = model_cfg.get("base_url")
        params: Dict[str, Any] = {
            "model": model_cfg.get("model", model_name),
            "base_url": base_url,
            "api_key": api_key,
        }
        if "temperature" in model_cfg:
            params["temperature"] = model_cfg["temperature"]
            
        # Merge model_kwargs
        if "model_kwargs" in model_cfg and "model_kwargs" in kwargs:
            merged = dict(model_cfg["model_kwargs"])
            merged.update(kwargs.pop("model_kwargs"))
            params["model_kwargs"] = merged
        elif "model_kwargs" in model_cfg:
            params["model_kwargs"] = dict(model_cfg["model_kwargs"])
            
        params.update(kwargs)
        return ChatOpenAI(**params)
        
    else:
        raise ValueError(f"Unsupported model provider '{provider}' for model '{model_name}'")
