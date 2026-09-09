@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
py -3.12 -m venv .venv
if errorlevel 1 (
  echo Install Python 3.12 with the Python launcher, then try again.
  exit /b 1
)
:install
".venv\Scripts\python.exe" -m pip install -r serving\requirements-test.txt
if errorlevel 1 exit /b 1
if not exist ".env" copy /y ".env.example" ".env" >nul
echo Setup complete. Add RUNPOD_API_KEY to .env, then run start-test.cmd when ready.
echo No model request was made.
