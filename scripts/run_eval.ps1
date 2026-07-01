# Project VARNA Evaluation Runbook
$env:PYTHONPATH="c:\IR-colorization-BAH2026"

Write-Host "==================================================" -ForegroundColor Green
Write-Host "STAGE 3: Running Scientific Evaluation" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
python cli.py evaluate

Write-Host "Evaluation complete. Metrics saved under experiments/varna_baseline/metrics.json" -ForegroundColor Green
