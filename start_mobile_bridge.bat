@echo off
title Sovereign Swarm Mobile Bridge
cd /d "%~dp0"
echo ========================================================
echo       LAUNCHING SOVEREIGN SWARM MOBILE BRIDGE
echo ========================================================
.venv\Scripts\python.exe start_mobile_bridge.py
pause
