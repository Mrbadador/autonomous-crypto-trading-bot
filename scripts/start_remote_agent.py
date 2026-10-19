"""
SOVEREIGN SWARM — ZERO-FRICTION REMOTE AGENT GATEWAY LAUNCHER
Spins up agent_mobile_bridge.py, resolves Cloudflare Tunnel, and prints terminal QR code.
"""

import os
import sys
import time
import re
import shutil
import urllib.request
import subprocess
import threading

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import qrcode
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

AGENT_PORT = int(os.getenv("AGENT_PORT", "8766"))
APP_PASSWORD = os.getenv("APP_PASSWORD")
if not APP_PASSWORD:
    raise EnvironmentError(
        "APP_PASSWORD environment variable is required. Set it in .env before starting the remote agent."
    )

CLOUDFLARED_BIN = os.path.join(BASE_DIR, "cloudflared.exe")

def ensure_cloudflared():
    if os.path.exists(CLOUDFLARED_BIN):
        return CLOUDFLARED_BIN
    found = shutil.which("cloudflared")
    if found:
        return found
    print("[-] Cloudflare binary not found.")
    return None

def print_qr_banner(url):
    print("\n" + "=" * 65)
    print("      SOVEREIGN SWARM — REMOTE AGENT GATEWAY READY")
    print("=" * 65)
    print(f"\n  Local Web UI  : http://localhost:{AGENT_PORT}")
    print(f"  Remote HTTPS  : {url}")
    print("  App Password  : (set in .env - never share this)\n")
    print("  [SCAN WITH PHONE CAMERA TO CHAT & EXECUTE DIRECTIVES]:\n")
    
    try:
        qr = qrcode.QRCode(border=2)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception as qre:
        print(f"  [QR Code render: {qre}]")
    
    print("\n  Tip: You can send tasks, inspect files, and execute commands from your phone.")
    print("=" * 65 + "\n")

def main():
    print("[*] Starting Sovereign Swarm Remote Agent Gateway...")
    
    # 1. Start FastAPI backend (bound to 127.0.0.1 for local security)
    web_cmd = [sys.executable, "-m", "uvicorn", "archive.legacy_agents.agent_mobile_bridge:app", "--host", "127.0.0.1", "--port", str(AGENT_PORT)]
    web_proc = subprocess.Popen(web_cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    time.sleep(1.5)
    
    print(f"[*] Remote agent gateway running on http://127.0.0.1:{AGENT_PORT}")
    print("[SECURITY] Cloudflare Tunnel disabled. Use VPN or SSH tunnel for remote access.")
    print("[SECURITY] App Password required from .env file.\n")

    # Local-only mode by default for security
    web_proc.wait()
    return

if __name__ == "__main__":
    main()
