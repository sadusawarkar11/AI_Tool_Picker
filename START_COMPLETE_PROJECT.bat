@echo off
title TOOLMATCHER AI - Complete Project Launcher
color 0A

echo ======================================================================
echo           🚀 TOOLMATCHER AI - COMPLETE PROJECT LAUNCHER
echo ======================================================================
echo.
echo Starting your complete project...
echo.

cd /d "%~dp0"

echo [1/3] Starting Flask Web Application...
echo.
start /B python start_app.py
timeout /t 8 /nobreak >nul

echo [2/3] Opening Web Browser...
echo.
start http://localhost:5000
timeout /t 2 /nobreak >nul

echo [3/3] Opening Jupyter Notebook...
echo.
start jupyter notebook toolmatcher_ai_advanced.ipynb

echo.
echo ======================================================================
echo                         ✅ PROJECT RUNNING!
echo ======================================================================
echo.
echo 📍 Flask Web App:      http://localhost:5000
echo 📊 Analytics:          http://localhost:5000/analytics  
echo 📓 Jupyter Notebook:   Opening in browser...
echo.
echo ⌨  Press Ctrl+C in this window to stop the Flask server
echo 📓 Close Jupyter notebook browser tab to stop Jupyter
echo.
echo ======================================================================
pause
