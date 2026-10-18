"""
SOVEREIGN SWARM — ZERO-FRICTION REMOTE MOBILE BRIDGE LAUNCHER
Spins up mobile_web_bridge.py, resolves Cloudflare Tunnel, and prints terminal QR code.
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

BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "8765"))
APP_PASSWORD = os.getenv("APP_PASSWORD")
if not APP_PASSWORD:
    raise EnvironmentError(
        "APP_PASSWORD environment variable is required. Set it in .env before starting the mobile bridge."
    )

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
CLOUDFLARED_BIN = os.path.join(BASE_DIR, "cloudflared.exe")

def ensure_cloudflared():
    if os.path.exists(CLOUDFLARED_BIN):
        return CLOUDFLARED_BIN
    
    # Check if in PATH
    found = shutil.which("cloudflared")
    if found:
        return found
        
    print("[*] Downloading official Cloudflare Tunnel binary (one-time setup)...")
    try:
        urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED_BIN)
        print("[+] Cloudflare Tunnel downloaded successfully.")
        return CLOUDFLARED_BIN
    except Exception as e:
        print(f"[-] Could not download cloudflared: {e}")
        return None

def print_qr_banner(url):
    print("\n" + "=" * 65)
    print("      SOVEREIGN SWARM — REMOTE MOBILE BRIDGE READY")
    print("=" * 65)
    print(f"\n  Local Access  : http://localhost:{BRIDGE_PORT}")
    print(f"  Remote HTTPS  : {url}")
    print(f"  App Password  : (set in .env - never share this)\n")
    print("  [SCAN WITH PHONE CAMERA FOR INSTANT ACCESS]:\n")
    
    try:
        qr = qrcode.QRCode(border=2)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception as qre:
        print(f"  [QR Code render skipped: {qre}]")
    
    print("\n  Tip: In iOS Safari tap 'Share' -> 'Add to Home Screen' for Native App Mode.")
    print("  Tip: In Android Chrome tap Menu -> 'Install App'.")
    print("=" * 65 + "\n")

def main():
    print("[*] Starting Sovereign Swarm Mobile Web Bridge...")
    
    # 1. Start FastAPI backend (bind to localhost only for security)
    web_cmd = [sys.executable, "-m", "uvicorn", "mobile_web_bridge:app", "--host", "127.0.0.1", "--port", str(BRIDGE_PORT)]
    web_proc = subprocess.Popen(web_cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    time.sleep(1.5)
    
    print(f"[*] Mobile bridge running on http://127.0.0.1:{BRIDGE_PORT}")
    print("[SECURITY] Cloudflare Tunnel disabled. Use VPN or SSH tunnel for remote access.")
    print(f"[SECURITY] App Password required from .env file.\n")
    
    # Wait for the web process
    web_proc.wait()
    return

    # Keep alive and pipe logs
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Halting Mobile Bridge and Tunnel...")
        web_proc.terminate()
        cf_proc.terminate()

if __name__ == "__main__":
    main()
