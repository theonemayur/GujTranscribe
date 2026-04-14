"""
GujTranscribe - PyInstaller Build Script
Run this to create the standalone executable
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

def build():
    print("=" * 50)
    print("  GujTranscribe - Build Script")
    print("=" * 50)
    print()
    
    base_dir = Path(__file__).parent
    source_dir = base_dir.parent / "GujTranscribe"
    temp_dir = base_dir / "build_temp"
    dist_dir = base_dir / "dist"
    
    print("[1/4] Cleaning previous build...")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    print("[2/4] Copying application files...")
    temp_dir.mkdir(exist_ok=True)
    
    app_source = source_dir / "gujarati_asr"
    app_dest = temp_dir / "gujarati_asr"
    shutil.copytree(app_source, app_dest)
    
    shutil.copy(source_dir / "requirements.txt", temp_dir)
    
    print("[3/4] Building executable...")
    print("      (This may take 10-20 minutes on first run)")
    print()
    
    os.chdir(temp_dir)
    
    hidden_imports = [
        "uvicorn", "fastapi", "transformers", "torch", "torch.nn",
        "librosa", "soundfile", "numpy", "scipy", "sklearn",
        "tokenizers", "huggingface_hub", "accelerate", "safetensors",
        "pydantic", "starlette", "anyio", "sniffio"
    ]
    
    pyinstaller_args = [
        "--name=GujTranscribe",
        "--onefile",
        "--console",
        "--clean",
        f'--add-data=gujarati_asr{os.pathsep}gujarati_asr',
    ]
    
    for imp in hidden_imports:
        pyinstaller_args.append(f"--hidden-import={imp}")
    
    pyinstaller_args.append("gujarati_asr/main.py")
    
    result = subprocess.run(
        ["pyinstaller"] + pyinstaller_args,
        capture_output=False
    )
    
    if result.returncode != 0:
        print("Build failed!")
        return False
    
    print()
    print("[4/4] Preparing output...")
    dist_dir.mkdir(exist_ok=True)
    exe_path = temp_dir / "dist" / "GujTranscribe.exe"
    
    if exe_path.exists():
        shutil.copy(exe_path, dist_dir)
        shutil.copy(temp_dir / "gujarati_asr" / "web", temp_dir / "web", dirs_exist_ok=True)
        shutil.copy(temp_dir / "gujarati_asr" / "data", temp_dir / "data", dirs_exist_ok=True)
        
        print()
        print("=" * 50)
        print("  Build Complete!")
        print("=" * 50)
        print()
        print(f"  Executable: {dist_dir / 'GujTranscribe.exe'}")
        print()
        print("  Users can double-click to run!")
        print()
        return True
    
    print("Build failed - executable not found")
    return False

if __name__ == "__main__":
    build()
    input("\nPress Enter to exit...")
