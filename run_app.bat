@echo off
REM ============================================
REM Batch file to run Flask Pothole Detection App
REM ============================================

echo.
echo =============================================
echo   Starting MLDL Pothole Detection App
echo =============================================
echo.

REM Activate the conda environment
echo [1/4] Activating conda environment MLDL...
call conda activate MLDL
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment MLDL
    echo Please make sure conda is installed and MLDL environment exists
    pause
    exit /b 1
)

echo.
echo [2/4] Starting Flask server...
echo.

REM Start Flask in a new command window so terminal stays open
start "Flask Server - MLDL Pothole Detection" cmd /k "python app.py"

REM Wait for Flask to start (give it time to initialize)
echo [3/4] Waiting for Flask to start...
timeout /t 5 /nobreak >nul

REM Open the default browser to the Flask app
echo [4/4] Opening browser...
start http://127.0.0.1:5000

echo.
echo =============================================
echo   Flask is running at http://127.0.0.1:5000
echo   The browser should have opened automatically
echo.
echo   To stop the server, close this window
echo   or press Ctrl+C in the Flask server window
echo =============================================
echo.
