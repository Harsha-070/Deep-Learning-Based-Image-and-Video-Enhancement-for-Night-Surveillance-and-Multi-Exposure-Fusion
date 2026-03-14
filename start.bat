@echo off
title Night Surveillance Enhancement System
cd /d "%~dp0"
echo.
echo  ============================================================
echo   Night Surveillance Enhancement System
echo  ============================================================
echo.
echo  Starting Streamlit...
echo  Open your browser to: http://localhost:8501
echo.
echo  Press Ctrl+C to stop the server.
echo  ============================================================
echo.
python -m streamlit run app.py --server.port 8501 --server.headless false
pause
