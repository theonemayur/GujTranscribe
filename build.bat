@echo off
echo ============================================
echo   GujTranscribe - Build Script
echo ============================================
echo.

echo [1/4] Installing build dependencies...
pip install pyinstaller

echo.
echo [2/4] Copying application files...
if not exist "build_temp" mkdir build_temp
xcopy /E /I /Y "..\GujTranscribe\gujarati_asr" "build_temp\gujarati_asr"
xcopy /E /I /Y "..\GujTranscribe\requirements.txt" "build_temp\"

echo.
echo [3/4] Building executable (this may take 10-20 minutes)...
cd build_temp

pyinstaller --onefile --name GujTranscribe --add-data "gujarati_asr;gujarati_asr" --hidden-import=uvicorn --hidden-import=fastapi --hidden-import=transformers --hidden-import=torch --hidden-import=librosa --hidden-import=soundfile main_launcher.py

echo.
echo [4/4] Copying executable to output folder...
if not exist "output" mkdir output
copy "dist\GujTranscribe.exe" "output\"

echo.
echo ============================================
echo   Build Complete!
echo ============================================
echo.
echo Executable: output\GujTranscribe.exe
echo.
echo Press any key to open the output folder...
pause >nul
explorer output

cd ..
