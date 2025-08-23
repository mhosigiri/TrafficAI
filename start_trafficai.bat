@echo off
echo ========================================
echo    TrafficAI Web Application
echo ========================================
echo.
echo Starting TrafficAI Web Application...
echo.
call .venv\Scripts\activate.bat
python web_app.py
pause
