@echo off
title Sovereign Swarm Remote Agent Gateway
cd /d "%~dp0"
echo ========================================================
echo       LAUNCHING REMOTE AGENT MOBILE GATEWAY
echo ========================================================
.venv\Scripts\python.exe start_remote_agent.py
pause
