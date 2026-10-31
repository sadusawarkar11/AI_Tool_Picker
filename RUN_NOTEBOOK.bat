@echo off
echo ======================================================================
echo        OPENING JUPYTER NOTEBOOK - TOOLMATCHER AI
echo ======================================================================
echo.
echo Opening your already-executed notebook in Jupyter...
echo.
echo The notebook will open with all previous outputs intact!
echo.
cd /d "%~dp0"
jupyter notebook toolmatcher_ai_advanced.ipynb
pause
