@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup-test.cmd first to create the Python environment.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "serving\launch_test.py" %*
pause
