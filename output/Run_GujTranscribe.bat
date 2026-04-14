@echo off
chcp 65001 >nul
echo ============================================
echo   GujTranscribe - Launcher
echo ============================================
echo.
echo Starting GujTranscribe...
echo.

cd /d "%~dp0"

start "" GujTranscribe.exe

timeout /t 3 /nobreak >nul

start http://localhost:8000

echo.
echo GujTranscribe is running!
echo Open http://localhost:8000 in your browser
echo.
echo Press Ctrl+C to stop the server
pause
