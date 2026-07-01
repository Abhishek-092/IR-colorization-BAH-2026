# Project VARNA End-to-End Phased Training Runbook
$env:PYTHONPATH="c:\IR-colorization-BAH2026"

Write-Host "==================================================" -ForegroundColor Green
Write-Host "STAGE 1: Training Backbone & Super-Resolution Head" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
python cli.py train-stage1

Write-Host "==================================================" -ForegroundColor Green
Write-Host "STAGE 2: Training Discretized Logistic Mixture Head" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
python cli.py train-stage2

Write-Host "Training complete. Checkpoints saved under experiments/varna_baseline/checkpoints/" -ForegroundColor Green
