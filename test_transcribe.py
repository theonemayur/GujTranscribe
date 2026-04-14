"""Test script for transcription"""
import requests
import os

SERVER = "http://localhost:8000"

# Test 1: Check server is running
print("1. Testing server status...")
try:
    r = requests.get(SERVER + "/", timeout=5)
    print(f"   Server status: {r.status_code}")
except Exception as e:
    print(f"   Server not running: {e}")
    print("   Starting server...")
    import subprocess
    subprocess.Popen(["python", "run_server.py"], cwd="gujarati_asr")
    import time
    time.sleep(10)
    r = requests.get(SERVER + "/", timeout=5)
    print(f"   Server now: {r.status_code}")

# Test 2: Test transcription endpoint
print("\n2. Testing /transcribe endpoint...")
test_file = r"C:\Users\PC\Desktop\GujTranscribe\Tester_1.mp3"
if os.path.exists(test_file):
    print(f"   File exists: {test_file}")
    with open(test_file, 'rb') as f:
        files = {'file': ('Tester_1.mp3', f, 'audio/mpeg')}
        try:
            r = requests.post(SERVER + "/transcribe", files=files, timeout=60)
            print(f"   Response: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                print(f"   Transcription: {data.get('transcription', 'N/A')[:100]}...")
            else:
                print(f"   Error: {r.text}")
        except Exception as e:
            print(f"   Error: {e}")
else:
    print(f"   File NOT found: {test_file}")
    # Try to find it
    import glob
    matches = glob.glob("**/Tester*.mp3", recursive=True)
    print(f"   Found: {matches}")
