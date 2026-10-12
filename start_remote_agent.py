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
APP_PASSWORD = os.getenv("APP_PASSWORD", "SovereignSwarm2026!")

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
    print(f"  App Password  : {APP_PASSWORD}\n")
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
    
    # 1. Start FastAPI backend
    web_cmd = [sys.executable, "-m", "uvicorn", "agent_mobile_bridge:app", "--host", "0.0.0.0", "--port", str(AGENT_PORT)]
    web_proc = subprocess.Popen(web_cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    time.sleep(1.5)
    
    # 2. Start Cloudflare Tunnel
    cf_bin = ensure_cloudflared()
    if not cf_bin:
        print(f"[!] Running in local-only mode: http://localhost:{AGENT_PORT}")
        web_proc.wait()
        return

    print("[*] Initializing secure Cloudflare Tunnel for Remote Agent...")
    cf_cmd = [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{AGENT_PORT}"]
    cf_proc = subprocess.Popen(cf_cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    tunnel_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    for line in iter(cf_proc.stdout.readline, ""):
        match = url_pattern.search(line)
        if match:
            tunnel_url = match.group(0)
            print_qr_banner(tunnel_url)
            break

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Halting Remote Agent Bridge...")
        web_proc.terminate()
        cf_proc.terminate()

if __name__ == "__main__":
    main()
