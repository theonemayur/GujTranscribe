@echo off
chcp 65001 >nul
echo ============================================
echo   GujTranscribe - Easy Build Script
echo ============================================
echo.
echo This will create a standalone executable
echo that users can run without installing Python.
echo.
echo The build may take 15-30 minutes.
echo.
echo Press Ctrl+C to cancel, or Enter to continue...
pause >nul

echo.
echo [1/5] Installing PyInstaller...
pip install pyinstaller -q

echo.
echo [2/5] Creating build folder...
if exist "build_temp" rmdir /s /q "build_temp"
mkdir "build_temp"

echo.
echo [3/5] Copying application...
xcopy /E /Q /Y "..\GujTranscribe\gujarati_asr" "build_temp\gujarati_asr\"
copy "..\GujTranscribe\requirements.txt" "build_temp\" >nul

echo.
echo [4/5] Building executable...
echo This will take a while. Please wait...
cd build_temp

pyinstaller --name GujTranscribe --onefile --console --clean ^
    --add-data "gujarati_asr;gujarati_asr" ^
    --hidden-import=uvicorn ^
    --hidden-import=fastapi ^
    --hidden-import=transformers ^
    --hidden-import=torch ^
    --hidden-import=librosa ^
    --hidden-import=soundfile ^
    --hidden-import=numpy ^
    --hidden-import=scipy ^
    --hidden-import=sklearn ^
    --hidden-import=tokenizers ^
    --hidden-import=huggingface_hub ^
    --hidden-import=accelerate ^
    --hidden-import=safetensors ^
    gujarati_asr/main.py

cd ..

echo.
echo [5/5] Moving to output folder...
if not exist "output" mkdir output
if exist "build_temp\dist\GujTranscribe.exe" (
    copy "build_temp\dist\GujTranscribe.exe" "output\"
    echo.
    echo ============================================
    echo   BUILD COMPLETE!
    echo ============================================
    echo.
    echo   Executable: output\GujTranscribe.exe
    echo.
    echo   Users just double-click to run!
    echo.
    echo   Press any key to open the folder...
    pause >nul
    explorer output
) else (
    echo.
    echo Build may have failed. Check for errors above.
    echo.
    pause
)
