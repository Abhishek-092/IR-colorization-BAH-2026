# Project SUTRAM Dataset Preparation Runbook
Write-Host "Starting input validation..." -ForegroundColor Cyan
python -c "
from data_pipeline.input_validator import validate_input_product
status, details = validate_input_product('input/LC09_L2SP_146044_20260701_20260701_02_T1')
if status:
    print('Input Validation PASSED')
else:
    print('Input Validation FAILED:', details)
"

Write-Host "Running custom patch generation..." -ForegroundColor Cyan
$env:PYTHONPATH="c:\IR-colorization-BAH2026"
python -m data_pipeline.prepare_dataset

Write-Host "Verifying patch alignment and co-registration..." -ForegroundColor Cyan
python -c "
from data_pipeline.alignment_checker import check_alignment_in_patches
check_alignment_in_patches('output/patches')
"
