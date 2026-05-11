cd C:\Users\user\Git\AIPA_Engine
$env:PYTHONIOENCODING = "utf-8"

$round = 1
while ($true) {
    Write-Host "========== Round $round ($(Get-Date -Format 'HH:mm:ss')) =========="
    C:\Users\user\miniconda3\envs\aipa_engine\python.exe data/scripts/pipeline.py --source naver
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Pipeline exited with error. API quota exhausted?"
        break
    }
    $round++
    Write-Host "Waiting 5 seconds..."
    Start-Sleep -Seconds 5
}
Write-Host "Pipeline stopped after $round rounds."
