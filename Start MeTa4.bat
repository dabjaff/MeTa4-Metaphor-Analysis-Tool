@echo off
cd /d "%~dp0"
set "PYTHONPATH=%CD%"
where py >nul 2>&1 && (py -m meta4.cli) || (python -m meta4.cli)
pause
