"""
start_tunnel.py — Mobile & Remote Public Access Hub for Sovereign Swarm HUD
Provides instant local Wi-Fi and global public HTTPS tunnel access to the Streamlit Dashboard.
"""

import os
import sys
import time
import socket
import shutil
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def main():
    local_ip = get_local_ip()
    port = 8501
    
    print("=" * 70)
    print("       📱 SOVEREIGN SWARM HUD — MOBILE & REMOTE ACCESS HUB")
    print("=" * 70)
    print()
    print("  [1] LOCAL WI-FI ACCESS (Same Network):")
    print(f"      👉 http://{local_ip}:{port}")
    print("      (Open this URL on your phone browser while connected to Wi-Fi)")
    print()
    print("  [2] LOCALHOST (This PC):")
    print(f"      👉 http://localhost:{port}")
    print()
    
    # Check for cloudflared
    cf_path = shutil.which("cloudflared")
    ngrok_path = shutil.which("ngrok")
    
    if cf_path:
        print("  [3] CLOUDFLARE QUICK TUNNEL (Global HTTPS Access from Anywhere):")
        print(f"      Launching cloudflared tunnel for port {port}...")
        try:
            cmd = [cf_path, "tunnel", "--url", f"http://localhost:{port}"]
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nTunnel stopped.")
    elif ngrok_path:
        print("  [3] NGROK TUNNEL (Global HTTPS Access from Anywhere):")
        print(f"      Launching ngrok http {port}...")
        try:
            cmd = [ngrok_path, "http", str(port)]
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nTunnel stopped.")
    else:
        print("  [3] GLOBAL ACCESS FROM ANYWHERE (No Wi-Fi Needed):")
        print("      To create a free instant HTTPS public URL on your phone:")
        print("      Option A (Cloudflare — Recommended, No Account Needed):")
        print("        1. Download 'cloudflared.exe' from https://github.com/cloudflare/cloudflared/releases")
        print(f"        2. Run: cloudflared.exe tunnel --url http://localhost:{port}")
        print("      Option B (ngrok):")
        print(f"        1. Run: ngrok http {port}")
    print()
    print("=" * 70)

if __name__ == "__main__":
    main()
