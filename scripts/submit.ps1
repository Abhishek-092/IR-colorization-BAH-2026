# Project VARNA Submission Runbook
$env:PYTHONPATH="c:\IR-colorization-BAH2026"

Write-Host "==================================================" -ForegroundColor Green
Write-Host "STAGE 4: Compiling Deliverable Submission Zip" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
python cli.py submit

Write-Host "Zip archive created successfully." -ForegroundColor Green
