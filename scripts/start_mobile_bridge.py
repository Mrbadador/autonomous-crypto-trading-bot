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
APP_PASSWORD = os.getenv("APP_PASSWORD", "SovereignSwarm2026!")

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
    print(f"  App Password  : {APP_PASSWORD}\n")
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
    
    # 1. Start FastAPI backend
    web_cmd = [sys.executable, "-m", "uvicorn", "mobile_web_bridge:app", "--host", "0.0.0.0", "--port", str(BRIDGE_PORT)]
    web_proc = subprocess.Popen(web_cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    time.sleep(1.5)
    
    # 2. Start Cloudflare Tunnel
    cf_bin = ensure_cloudflared()
    if not cf_bin:
        print(f"[!] Running in local-only mode: http://localhost:{BRIDGE_PORT}")
        web_proc.wait()
        return

    print("[*] Initializing secure encrypted Cloudflare Tunnel...")
    cf_cmd = [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{BRIDGE_PORT}"]
    cf_proc = subprocess.Popen(cf_cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    tunnel_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    for line in iter(cf_proc.stdout.readline, ""):
        match = url_pattern.search(line)
        if match:
            tunnel_url = match.group(0)
            print_qr_banner(tunnel_url)
            break

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
