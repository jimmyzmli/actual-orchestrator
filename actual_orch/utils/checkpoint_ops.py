import os
import glob
import sqlite3
from typing import List

def clear_all_checkpoints(db_path: str, sources: List[str] = None) -> int:
    """
    Clears all checkpoint database files matching the path.
    If sources is provided, instead of deleting the file, it connects to sqlite and deletes matching rows.
    Returns the number of files/rows deleted.
    """
    if not sources or ("email" in sources and "gdrive" in sources):
        files = glob.glob(f"{db_path}*")
        count = 0
        if files:
            for f in files:
                os.remove(f)
                count += 1
        return count
    
    if not os.path.exists(db_path):
        return 0
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    deleted = 0
    
    if "email" in sources:
        # Email thread_ids do not contain '/'
        c.execute("DELETE FROM checkpoints WHERE thread_id NOT LIKE '%/%'")
        c.execute("DELETE FROM writes WHERE thread_id NOT LIKE '%/%'")
        deleted += c.rowcount
        
    if "gdrive" in sources:
        # GDrive thread_ids contain '/'
        c.execute("DELETE FROM checkpoints WHERE thread_id LIKE '%/%'")
        c.execute("DELETE FROM writes WHERE thread_id LIKE '%/%'")
        deleted += c.rowcount
        
    conn.commit()
    conn.close()
    return deleted

def purge_checkpoint_threads(db_path: str, thread_ids: List[str]) -> int:
    """
    Deletes specific thread IDs from the checkpoint database.
    Returns the number of rows deleted.
    """
    if not os.path.exists(db_path):
        return 0
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    deleted = 0
    
    for thread_id in thread_ids:
        c.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        c.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        deleted += c.rowcount
        
    conn.commit()
    conn.close()
    
    return deleted
