import os
from datetime import datetime

LOGS_FILE = os.path.join(os.path.dirname(__file__), "logs")

def write_logs(text):
    if not os.path.exists(LOGS_FILE):
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOGS_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n\n")
