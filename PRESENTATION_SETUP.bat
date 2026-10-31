@echo off
echo.
echo ================================================================================
echo                   TOOLMATCHER AI - PRESENTATION SETUP
echo ================================================================================
echo.
echo This will open:
echo   1. Flask Web App (Frontend)
echo   2. Jupyter Notebook (ML Training Process)
echo.
echo ================================================================================
echo.
pause

echo.
echo [Step 1/2] Starting Flask Server...
echo ================================================================================
start "ToolMatcher AI - Flask Server" cmd /k "cd /d "%~dp0" && python start_app.py"
echo.
echo Waiting 5 seconds for server to start...
timeout /t 5 /nobreak > nul

echo.
echo [Step 2/2] Opening Jupyter Notebook...
echo ================================================================================
start "ToolMatcher AI - Jupyter" cmd /k "cd /d "%~dp0" && jupyter notebook toolmatcher_ai_advanced.ipynb"
echo.
echo Waiting 3 seconds for Jupyter to start...
timeout /t 3 /nobreak > nul

echo.
echo ================================================================================
echo                          SETUP COMPLETE!
echo ================================================================================
echo.
echo Your browser will open with:
echo   - Tab 1: Jupyter Notebook (already executed with all outputs)
echo   - Tab 2: Web App at http://localhost:5000
echo.
echo Two terminal windows are open in the background:
echo   - Flask Server (don't close this!)
echo   - Jupyter Notebook (don't close this!)
echo.
echo ================================================================================
echo                    READY FOR PRESENTATION!
echo ================================================================================
echo.
echo When finished presenting:
echo   1. Close this window
echo   2. Close the Flask Server window
echo   3. Close the Jupyter window (press Ctrl+C twice)
echo.
echo ================================================================================
pause
