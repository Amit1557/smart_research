@echo off
echo ========================================================
echo   SmartDoc RAG: Machine Learning, RAG ^& Gen AI Assistant
echo ========================================================
echo.

cd /d "%~dp0backend"

echo Checking Python dependencies...
python -m pip install -r requirements.txt

echo.
echo Starting FastAPI Backend Server on http://localhost:8000 ...
echo Open your browser to: http://localhost:8000
echo.
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
