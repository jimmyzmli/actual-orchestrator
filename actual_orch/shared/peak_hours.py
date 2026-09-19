import json
import os
from datetime import datetime

def check_peak_hours(model_name: str = "deepseek") -> bool:
    if os.environ.get("ALLOW_PEAK_HOURS") == "1":
        return False
        
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/peak_hours.json"))
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "config", "peak_hours.json")
    if not os.path.exists(config_path):
        return False
        
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            
        tz_name = config.get("timezone", "America/New_York")
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(tz_name)
            now = datetime.now(tz)
            is_dst = now.dst().total_seconds() != 0 if now.dst() else False
        except (ImportError, Exception):
            import time
            now = datetime.now()
            is_dst = time.localtime().tm_isdst > 0
            
        current_time = now.time()
        
        models_config = config.get("models", {})
        schedules = models_config.get(model_name, [])
        if not schedules:
            return False
            
        for schedule in schedules:
            if schedule.get("is_dst") == is_dst:
                for window in schedule.get("windows", []):
                    start_str = window.get("start")
                    end_str = window.get("end")
                    if start_str and end_str:
                        start_h, start_m = map(int, start_str.split(":"))
                        end_h, end_m = map(int, end_str.split(":"))
                        from datetime import time as dt_time
                        start_time = dt_time(start_h, start_m)
                        end_time = dt_time(end_h, end_m)
                        
                        if start_time <= end_time:
                            if start_time <= current_time <= end_time:
                                return True
                        else:
                            if current_time >= start_time or current_time <= end_time:
                                return True
                                
    except Exception as e:
        print(f"Error checking peak hours: {e}")
        
    return False

def print_peak_hours_warning(model_name: str = "deepseek"):
    # Check if any configured agent uses a model with this peak hours schedule
    try:
        from actual_orch.shared.utils import get_config_path
        from actual_orch.shared.models import get_model_config
        agents_path = get_config_path("agents.json")
        if os.path.exists(agents_path):
            with open(agents_path, "r") as f:
                agents_cfg = json.load(f)
            uses_schedule = False
            for task, task_cfg in agents_cfg.items():
                if isinstance(task_cfg, dict):
                    for att in task_cfg.get("attempts", []):
                        m = att.get("model")
                        if m and get_model_config(m).get("peak_hours") == model_name:
                            uses_schedule = True
                            break
                if uses_schedule:
                    break
            if not uses_schedule:
                return
    except Exception:
        pass

    if check_peak_hours(model_name):
        print(f"\n⚠️  WARNING: Currently in Peak Hours for {model_name}.")
        print(f"Processing with {model_name} during this time may experience API limits or rate throttling.")
        print("Agent calls will pause for confirmation unless --allow-peak-hours is passed.\n")

def pause_if_peak_hours(model_name: str = "deepseek"):
    if check_peak_hours(model_name):
        import sys
        if not sys.stdin or not hasattr(sys.stdin, "isatty") or not sys.stdin.isatty():
            print(f"\n⚠️  Currently in peak hours for {model_name} (non-interactive session; proceeding automatically).")
            return
        try:
            input(f"\n⚠️  Currently in peak hours for {model_name}! Press Enter to run agent/LLM, or Ctrl+C to abort...")
        except (KeyboardInterrupt, EOFError, OSError, IOError) as e:
            if isinstance(e, KeyboardInterrupt):
                import sys
                print("\nAborted by user.")
                sys.exit(1)
            else:
                print(f"\n⚠️  Currently in peak hours for {model_name} (stdin unavailable; proceeding).")
