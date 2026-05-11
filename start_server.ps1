# Kill existing process on port 8080
$port = 8080
$connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
if ($connections) {
    $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($pid in $pids) {
        $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "[!] Killing process on port ${port}: $($proc.ProcessName) (PID: $pid)" -ForegroundColor Yellow
            Stop-Process -Id $pid -Force
        }
    }
    Start-Sleep -Seconds 1
    Write-Host "[OK] Port cleared" -ForegroundColor Green
} else {
    Write-Host "[OK] Port $port available" -ForegroundColor Green
}

# Start server
Write-Host ""
Write-Host "=== AIPA Engine Server Starting ===" -ForegroundColor Cyan
cd C:\Users\user\Git\AIPA_Engine
$env:PYTHONIOENCODING = "utf-8"
# --reload 제거: Windows에서 Ctrl+C 종료 불가 버그 방지
# 코드 수정 후 서버 재시작 필요 시: Ctrl+C -> .\start_server.ps1
C:\Users\user\miniconda3\envs\aipa_engine\python.exe -m uvicorn aipa_engine.main:app --host 0.0.0.0 --port 8080
