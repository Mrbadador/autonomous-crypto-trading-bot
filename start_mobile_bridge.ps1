# Sovereign Swarm Mobile Bridge Launcher
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "      LAUNCHING SOVEREIGN SWARM MOBILE BRIDGE" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
& ".venv\Scripts\python.exe" "start_mobile_bridge.py"
