@echo off
rem Double-click to vet synthetic rewrites (see src/radio_ready/vet.py).
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
python -m radio_ready.vet %*
if errorlevel 1 pause
