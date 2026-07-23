from typing import Iterator, Tuple, Set, Callable, List, Optional
import actual_orch.clients.cache as cache_module

def export_cache(prefix: str, cache_type: str = None) -> Iterator[Tuple[str, str, str]]:
    """
    Yields (identifier, cache_type, value) for keys matching prefix*.
    """
    cache_module._init_cache()
        
    client = cache_module._client
    keys = client.keys(f"{prefix}*")
    
    if cache_type:
        type_bytes = f":{cache_type}".encode("utf-8")
        type_str = f":{cache_type}"
        keys = [k for k in keys if (type_bytes in k if isinstance(k, bytes) else type_str in k)]
        
    for key in keys:
        key_str = key.decode("utf-8") if isinstance(key, bytes) else key
        val = client.get(key)
        if not val:
            continue
            
        without_prefix = key_str[len(prefix):]
        parts = without_prefix.rsplit(":", 1)
        
        if len(parts) == 2:
            identifier, ctype = parts
        else:
            identifier = without_prefix
            ctype = cache_type or "unknown"
            
        yield identifier, ctype, val


def prune_orphaned_keys(prefix: str, valid_identifiers: Set[str], extract_identifier_cb: Optional[Callable[[str], Optional[str]]] = None) -> int:
    """
    Deletes keys matching prefix* where the extracted identifier is not in valid_identifiers.
    """
    cache_module._init_cache()
        
    client = cache_module._client
    keys = client.keys(f"{prefix}*")
    pruned = 0
    
    for key in keys:
        key_str = key.decode("utf-8") if isinstance(key, bytes) else key
        if extract_identifier_cb:
            identifier = extract_identifier_cb(key_str)
        else:
            without_prefix = key_str[len(prefix):]
            identifier = without_prefix.rsplit(":", 1)[0]
            
        if identifier is not None and identifier not in valid_identifiers:
            client.delete(key)
            pruned += 1
            
    return pruned


def clear_redis_cache(prefix: str, cache_type: str = None) -> int:
    """
    Deletes all keys matching the prefix and cache_type pattern.
    """
    cache_module._init_cache()
        
    client = cache_module._client
    pattern = f"{prefix}*"
    if cache_type:
        pattern = f"{prefix}*:{cache_type}"
        
    keys = client.keys(pattern)
    if keys:
        client.delete(*keys)
        return len(keys)
    return 0


def purge_redis_keys(prefix: str, identifiers: List[str], cache_type: str = None) -> int:
    """
    Deletes specific identifiers from Redis. Try exact match, then pattern match.
    """
    cache_module._init_cache()
        
    client = cache_module._client
    deleted = 0
    
    for identifier in identifiers:
        if client.exists(identifier):
            client.delete(identifier)
            deleted += 1
            continue
            
        pattern = f"{prefix}*{identifier}*"
        keys = client.keys(pattern)
        
        if cache_type:
            type_bytes = f":{cache_type}".encode("utf-8")
            type_str = f":{cache_type}"
            keys = [k for k in keys if (type_bytes in k if isinstance(k, bytes) else type_str in k)]
            
        if keys:
            client.delete(*keys)
            deleted += len(keys)
            
    return deleted
