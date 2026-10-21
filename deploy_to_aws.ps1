<#
.SYNOPSIS
    Syncs local changes from your PC to the AWS EC2 trading bot instance and triggers clean fleet recycling.
.USAGE
    .\deploy_to_aws.ps1
    .\deploy_to_aws.ps1 -Restart
#>
param(
    [string]$ServerIp = "13.53.41.118",
    [string]$KeyPath = "$env:USERPROFILE\.ssh\aws_trading_key.pem",
    [switch]$Restart = $false
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " 🚀 SYNCING LOCAL CHANGES TO AWS EC2" -ForegroundColor Cyan
Write-Host " Target: ubuntu@$ServerIp" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Create a clean archive of git-tracked files
$ArchiveName = "sync_bundle.tar.gz"
Write-Host "`n📦 Packaging current codebase..." -ForegroundColor Yellow
git archive --format=tar.gz -o $ArchiveName HEAD

try {
    # 2. Transfer the archive to EC2
    Write-Host "📤 Uploading package to EC2..." -ForegroundColor Yellow
    scp -i $KeyPath $ArchiveName ubuntu@${ServerIp}:~/llama_trading_bot/

    # 3. Unpack on EC2
    Write-Host "📂 Unpacking updates on server..." -ForegroundColor Yellow
    ssh -i $KeyPath ubuntu@$ServerIp "cd ~/llama_trading_bot && tar -xzf sync_bundle.tar.gz && rm sync_bundle.tar.gz"

    if ($Restart) {
        Write-Host "🔄 Restarting fleet orchestrator inside tmux..." -ForegroundColor Yellow
        ssh -i $KeyPath ubuntu@$ServerIp "tmux send-keys -t trading C-c; sleep 2; tmux send-keys -t trading './venv/bin/python start_system.py' C-m"
    } else {
        Write-Host "⚡ Note: start_system.py watches modified files, but you can pass -Restart to bounce the fleet anytime." -ForegroundColor DarkGray
    }

    Write-Host "`n✅ SYNC COMPLETE! Your EC2 instance is now up-to-date with local code." -ForegroundColor Green
} finally {
    if (Test-Path $ArchiveName) {
        Remove-Item $ArchiveName -Force
    }
}
