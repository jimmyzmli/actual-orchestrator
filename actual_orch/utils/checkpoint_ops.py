import os
import glob
import sqlite3
from typing import List

def clear_all_checkpoints(db_path: str) -> int:
    """
    Clears all checkpoint database files matching the path.
    Returns the number of files deleted.
    """
    files = glob.glob(f"{db_path}*")
    count = 0
    if files:
        for f in files:
            os.remove(f)
            count += 1
    return count

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
