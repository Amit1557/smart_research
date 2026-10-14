@echo off
echo ========================================================
echo   SmartDoc RAG: Machine Learning, RAG ^& Gen AI Assistant
echo ========================================================
echo.

cd /d "%~dp0"

:: Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
)

echo [1/2] Activating Virtual Environment (venv)...
call venv\Scripts\activate.bat

echo [2/2] Starting FastAPI Backend on http://localhost:8000 ...
echo Open your browser to: http://localhost:8000
echo.
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
