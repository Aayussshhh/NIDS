@echo off
REM ============================================================================
REM Convenience launcher for the Hybrid NIDS backend (Windows).
REM
REM Opens two terminals:
REM   1. Backend (FastAPI) - must run as Administrator for packet capture
REM   2. Frontend (Vite dev server) - regular user is fine
REM
REM Usage:   run.bat
REM ============================================================================

echo.
echo Starting Hybrid NIDS Operations Console...
echo.

REM --- Start the Python backend in a new admin Command Prompt
echo [1/2] Launching backend (FastAPI)...
powershell -Command "Start-Process cmd -Verb RunAs -ArgumentList '/k cd /d %~dp0 && .venv\Scripts\activate && python -m backend.api.main'"

REM Give the backend a moment to start before launching the frontend.
timeout /t 3 /nobreak >nul

REM --- Start the React frontend (non-admin)
echo [2/2] Launching frontend (Vite)...
start cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Two terminals have been opened.
echo   - Backend: http://localhost:8000
echo   - Frontend: http://localhost:5173
echo.
echo Open http://localhost:5173 in your browser.
