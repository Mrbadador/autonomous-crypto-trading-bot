@echo off
title SOVEREIGN FIRM V28.1 - MASTER TERMINAL
color 0A
echo ===================================================
echo   IGNITING SOVEREIGN QUANTITATIVE ENGINE...
echo ===================================================
cd /d "%~dp0"

REM Activate the project virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [ENV] Virtual environment activated.
) else (
    echo [WARN] No .venv found. Using system Python.
)

python start_system.py
pause
