# Sovereign Swarm Remote Agent Gateway Launcher
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "      LAUNCHING REMOTE AGENT MOBILE GATEWAY" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
& ".venv\Scripts\python.exe" "start_remote_agent.py"
