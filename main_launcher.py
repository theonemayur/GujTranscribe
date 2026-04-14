"""
GujTranscribe - Standalone Executable Launcher
"""
import sys
import os
import subprocess
import webbrowser
import time
import threading
from pathlib import Path

def get_script_dir():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent

def main():
    print("=" * 50)
    print("  GujTranscribe - Gujarati ASR")
    print("=" * 50)
    print()
    
    base_dir = get_script_dir()
    web_dir = base_dir / "web"
    exe_path = sys.executable
    
    print("[1/3] Starting server...")
    
    server_script = base_dir / "gujarati_asr" / "main.py"
    
    if not server_script.exists():
        server_script = base_dir / "main.py"
    
    if server_script.exists():
        proc = subprocess.Popen(
            [exe_path, str(server_script)],
            cwd=str(base_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    else:
        print("Error: Server script not found!")
        input("Press Enter to exit...")
        return
    
    print("[2/3] Waiting for server to start...")
    time.sleep(5)
    
    print("[3/3] Opening browser...")
    time.sleep(2)
    webbrowser.open("http://localhost:8000")
    
    print()
    print("=" * 50)
    print("  GujTranscribe is running!")
    print("  Open http://localhost:8000 in your browser")
    print("  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping server...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
