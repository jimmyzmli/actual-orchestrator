"""
Backward-compatible shim — delegates to the new `update.py receipt process` command
or prints a warning if someone runs this directly.
"""

import sys
import subprocess
import os

if __name__ == "__main__":
    print("WARNING: actual-orch has been refactored into a module.")
    print("Please use `update.py receipt <command>` from actual-utils instead.")
    print("Redirecting to `update.py receipt process` for backward compatibility...\n")
    
    utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "actual-utils")
    update_script = os.path.join(utils_dir, "update.py")
    
    if os.path.exists(update_script):
        cmd = [sys.executable, update_script, "receipt"]
        if len(sys.argv) > 1:
            cmd.extend(sys.argv[1:])
        else:
            cmd.append("process")
        sys.exit(subprocess.call(cmd, cwd=utils_dir))
    else:
        print(f"Could not find {update_script}")
        sys.exit(1)
