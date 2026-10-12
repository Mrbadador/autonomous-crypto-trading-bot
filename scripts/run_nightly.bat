@echo off
:: Nightly Sequence Wrapper for Windows Task Scheduler
cd /d "%~dp0.."
set LOGDIR=logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set MASTERLOG=%LOGDIR%\nightly_master.log

if exist ".venv\Scripts\python.exe" (
    set "PYBIN=.venv\Scripts\python.exe"
) else (
    set "PYBIN=python"
)

echo ================================================= >> "%MASTERLOG%"
echo [%date% %time%] STARTING NIGHTLY SEQUENCE >> "%MASTERLOG%"
echo ================================================= >> "%MASTERLOG%"

echo [%date% %time%] [1/2] Launching Nightly Auditor... >> "%MASTERLOG%"
%PYBIN% scripts\nightly_auditor.py >> "%MASTERLOG%" 2>&1
set AUDIT_RC=%errorlevel%
echo [%date% %time%] [1/2] Auditor exited with code: %AUDIT_RC% >> "%MASTERLOG%"

echo [%date% %time%] [2/2] Launching Nightly Trainer... >> "%MASTERLOG%"
%PYBIN% scripts\nightly_trainer.py >> "%MASTERLOG%" 2>&1
set TRAIN_RC=%errorlevel%
echo [%date% %time%] [2/2] Trainer exited with code: %TRAIN_RC% >> "%MASTERLOG%"

echo [%date% %time%] NIGHTLY SEQUENCE COMPLETE >> "%MASTERLOG%"
echo ================================================= >> "%MASTERLOG%"
exit /b %TRAIN_RC%
