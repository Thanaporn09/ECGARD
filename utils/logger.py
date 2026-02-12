import os
import json
import datetime
import atexit

class JSONLogger:
    """Logger that writes logs in JSON format (file only)."""

    def __init__(self, out_dir=None):
        # Determine final log directory
        base_dir = out_dir or "logs"
        log_dir = os.path.join(base_dir, "logs") if os.path.isdir(base_dir) else base_dir
        os.makedirs(log_dir, exist_ok=True)

        # Timestamped log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{timestamp}.log")

        # Create file and announce
        self.file = open(self.log_path, "a", buffering=1)
        print(f"🧾 Logging to file: {self.log_path}")

        # Ensure closure on exit
        atexit.register(self.close)

    def log(self, data: dict):
        """Write JSON-formatted log entry (no console print)."""
        data["time"] = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        json_line = json.dumps(data, ensure_ascii=False)
        self.file.write(json_line + "\n")

    def log_env(self, env_info):
        """Log environment or config info at the beginning."""
        self.log({"phase": "env_info", "details": env_info})

    def close(self):
        if not self.file.closed:
            self.file.close()
            print(f"✅ Log file saved at: {self.log_path}")
