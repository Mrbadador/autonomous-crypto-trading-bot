import webbrowser
import time
import os

url = "http://localhost:8501"
print(f"[NATIVE OS LAUNCH] Blasting command to default system browser for: {url}")
# Enforce direct shell invocation to push it onto the interactive display layer
os.system(f"start chrome {url}" if os.name == 'nt' else f"open {url}")
time.sleep(2)
