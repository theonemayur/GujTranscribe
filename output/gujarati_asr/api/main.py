"""
FastAPI backend for Gujarati + Gujlish ASR.
Endpoints:
  - POST /transcribe: upload audio file, get transcription
  - POST /subtitle: upload audio file, get SRT subtitles
  - WebSocket /ws/transcribe: real-time streaming transcription
"""
import os
import json
import asyncio
import tempfile
import sys
import threading
import subprocess
import csv
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional
from datetime import datetime
import uuid
import base64

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import librosa
import soundfile as sf
import numpy as np
import re
from api.transliteration import GujlishConverter

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = os.getenv("MODEL_DIR", str(BASE_DIR / "models" / "whisper_story_v2"))
PRETRAINED_MODEL = os.getenv("PRETRAINED_MODEL", "vasista22/whisper-gujarati-small")
DATA_DIR = BASE_DIR / "data" / "submitted_samples"
SUPPORTED_AUDIO_EXTENSIONS = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm', '.wma', '.aac')
CUDA_AVAILABLE = torch.cuda.is_available()
preferred_device = os.getenv("ASR_DEVICE_MODE", "auto").lower()  # auto | cpu | gpu
active_device = "cuda" if CUDA_AVAILABLE and preferred_device != "cpu" else "cpu"
PROFILE_FILE = DATA_DIR / "profile.json"
COLLECTED_DATA_FILE = DATA_DIR / "collected_data.jsonl"
COLLECTED_AUDIO_DIR = DATA_DIR / "collected_audio"

DATA_DIR.mkdir(parents=True, exist_ok=True)
COLLECTED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

transliteration_converter = GujlishConverter()

WORD_CORRECTIONS_FILE = DATA_DIR / "word_corrections.json"
word_corrections = {}

def load_word_corrections():
    global word_corrections
    if WORD_CORRECTIONS_FILE.exists():
        with open(WORD_CORRECTIONS_FILE, 'r', encoding='utf-8') as f:
            word_corrections = json.load(f)

def save_word_correction(wrong_word, correct_word):
    word_corrections[wrong_word.lower().strip()] = correct_word.strip()
    with open(WORD_CORRECTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(word_corrections, f, ensure_ascii=False)

def apply_corrections(text):
    words = text.split()
    corrected = []
    for word in words:
        lower = word.lower()
        if lower in word_corrections:
            corrected.append(word_corrections[lower])
        else:
            corrected.append(word)
    return ' '.join(corrected)

load_word_corrections()

print(f"Loading ASR model from HuggingFace on {active_device} (preferred={preferred_device})...", flush=True)

asr_pipeline = None

def _resolve_pipeline_device() -> int:
    if preferred_device == "cpu":
        return -1
    if preferred_device == "gpu":
        if not CUDA_AVAILABLE:
            raise RuntimeError("GPU requested but CUDA is not available on this machine")
        return 0
    return 0 if CUDA_AVAILABLE else -1

def load_model():
    global asr_pipeline, active_device
    device_index = _resolve_pipeline_device()
    active_device = "cuda" if device_index >= 0 else "cpu"
    try:
        print(f"Trying pre-trained Gujarati model: {PRETRAINED_MODEL} on {active_device}", flush=True)
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=PRETRAINED_MODEL,
            device=device_index
        )
        print(f"Loaded {PRETRAINED_MODEL} on {active_device}!", flush=True)
        return True
    except Exception as e:
        print(f"Failed to load pre-trained model: {e}", flush=True)

    try:
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=device_index
        )
        print(f"Loaded openai/whisper-tiny fallback on {active_device}!", flush=True)
        return True
    except Exception as e2:
        print(f"All models failed: {e2}", flush=True)
        return False

model_loaded = load_model()

UPDATE_MODEL_DIR = BASE_DIR / "models" / "whisper_submitted_update"
UPDATE_LOG_FILE = DATA_DIR / "model_update.log"
update_lock = threading.Lock()
model_update_status = {
    "status": "idle",  # idle | running | success | error
    "message": "No update started yet.",
    "started_at": None,
    "finished_at": None,
    "samples_count": 0,
    "audio_samples_count": 0,
    "output_model_dir": str(UPDATE_MODEL_DIR),
}

try:
    print("Checking for better translation model...", flush=True)
    try:
        transliterator = pipeline("translation", model="cointegrated/rut5-base-multilingual-greek", device=-1)
        print("Translation model loaded!", flush=True)
    except:
        transliterator = None
        print("Using fallback conversion", flush=True)
except Exception as e:
    print(f"Translation not available: {e}", flush=True)
    transliterator = None

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Gujarati + Gujlish ASR API")

# Mount static files for web UI
WEB_DIR = BASE_DIR / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

@app.get("/")
async def root():
    dashboard_path = WEB_DIR / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(str(dashboard_path))
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Gujarati + Gujlish ASR API is running."}


@app.get("/submit.html")
async def submit_page():
    submit_path = WEB_DIR / "submit.html"
    if submit_path.exists():
        return FileResponse(str(submit_path))
    return {"message": "Submit page not found"}


@app.get("/sync.html")
async def sync_page():
    sync_path = WEB_DIR / "sync.html"
    if sync_path.exists():
        return FileResponse(str(sync_path))
    return {"message": "Sync page not found"}


@app.get("/convert.html")
async def convert_page():
    convert_path = WEB_DIR / "convert.html"
    if convert_path.exists():
        return FileResponse(str(convert_path))
    return {"message": "Convert page not found"}


# ----------------------------
# Helper functions
# ----------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

manager = ConnectionManager()


def load_audio(file_path: str, target_sr: int = 16000):
    """Load audio file, resample to target_sr, mono."""
    print(f"Loading audio from {file_path}...", flush=True)
    # 1) Most robust for compressed formats (ogg/mp3/m4a/etc.)
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        print(f"Loaded with librosa: sr={sr}, len={len(audio)}", flush=True)
        return audio, target_sr
    except Exception as e1:
        print(f"Librosa load failed: {e1}", flush=True)

    # 2) Try soundfile (works well for wav/flac/ogg on most systems)
    try:
        audio, sr = sf.read(file_path, always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        print(f"Loaded with soundfile: sr={sr}, len={len(audio)}", flush=True)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio, target_sr
    except Exception as e2:
        print(f"Soundfile load failed: {e2}", flush=True)

    # 3) Final fallback for PCM wav files
    try:
        from scipy.io import wavfile
        sr, audio = wavfile.read(file_path)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size and np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0
        print(f"Loaded with scipy wavfile: sr={sr}, len={len(audio)}", flush=True)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio, target_sr
    except Exception as e3:
        raise ValueError(f"Cannot read audio file. librosa={e1}; soundfile={e2}; scipy={e3}")


def transcribe_audio(audio_array, sampling_rate: int = 16000):
    """Run Whisper transcription on audio array."""
    if asr_pipeline is not None:
        try:
            result = asr_pipeline(audio_array, return_timestamps=False)
        except:
            result = asr_pipeline(audio_array)
        text = result.get("text", "")
        text = apply_corrections(text)
        return text
    
    return ""


def transcribe_with_timestamps(audio_array, sampling_rate: int = 16000, chunk_length_s: float = 5.0):
    """
    Transcribe long audio by chunking and return segments with timestamps.
    """
    total_duration = len(audio_array) / sampling_rate
    if total_duration <= chunk_length_s:
        if asr_pipeline is not None:
            try:
                result = asr_pipeline(audio_array, return_timestamps=True)
            except:
                result = asr_pipeline(audio_array)
            text = result.get("text", "")
            chunks = result.get("chunks", [])
            if chunks:
                return [{"start": c.get("timestamp", [0, 0])[0], "end": c.get("timestamp", [0, 0])[1], "text": c.get("text", "")} for c in chunks]
            return [{"start": 0.0, "end": total_duration, "text": text}]
        return [{"start": 0.0, "end": total_duration, "text": ""}]
    
    overlap_s = 1.0
    chunk_samples = int(chunk_length_s * sampling_rate)
    step = chunk_samples - int(overlap_s * sampling_rate)
    
    segments = []
    start_time = 0.0
    while start_time < total_duration:
        end_time = min(start_time + chunk_length_s, total_duration)
        start_sample = int(start_time * sampling_rate)
        end_sample = int(end_time * sampling_rate)
        chunk = audio_array[start_sample:end_sample]
        
        text = transcribe_audio(chunk, sampling_rate)
        segments.append({
            "start": start_time,
            "end": end_time,
            "text": text.strip()
        })
        
        start_time += chunk_length_s - overlap_s
        if start_time >= total_duration:
            break
    
    return segments


# ----------------------------
# API Endpoints
# ----------------------------
class TranslateRequest(BaseModel):
    text: str
    source: str = "gujlish"
    target: str = "gujarati"

class RuntimeConfigRequest(BaseModel):
    mode: str = "auto"  # auto | cpu | gpu
    reload_model: bool = True

class ProfileRequest(BaseModel):
    name: str = "User"
    organization: str = ""
    language_focus: str = "gujarati"
    goal: str = "Improve Gujarati ASR quality"

class CollectRequest(BaseModel):
    urls: list[str]
    mode: str = "web_text"  # web_text | youtube_audio
    language: str = "gujarati"

@app.post("/translate")
async def translate_text(request: TranslateRequest):
    """
    Convert Gujlish to Gujarati using phonetic transliteration.
    """
    text = request.text.strip()
    if not text:
        return {"translation": ""}
    
    try:
        if request.source == "gujlish" and request.target == "gujarati":
            result = transliteration_converter.convert(text)
            return {"translation": result}
        elif request.source == "gujarati" and request.target == "gujlish":
            from api.transliteration import gujarati_to_gujlish
            result = gujarati_to_gujlish(text)
            return {"translation": result}
        else:
            return {"translation": text}
    except Exception as e:
        return {"translation": text, "error": str(e)}


@app.get("/runtime-config")
async def get_runtime_config():
    return {
        "preferred_mode": preferred_device,
        "active_device": active_device,
        "cuda_available": CUDA_AVAILABLE,
        "model_loaded": model_loaded,
        "model_name": PRETRAINED_MODEL if model_loaded else "none",
    }


@app.post("/runtime-config")
async def set_runtime_config(request: RuntimeConfigRequest):
    global preferred_device, model_loaded
    mode = (request.mode or "auto").lower().strip()
    if mode not in ("auto", "cpu", "gpu"):
        raise HTTPException(status_code=400, detail="mode must be one of: auto, cpu, gpu")
    if mode == "gpu" and not CUDA_AVAILABLE:
        raise HTTPException(status_code=400, detail="GPU mode requested, but CUDA is not available")
    preferred_device = mode
    if request.reload_model:
        model_loaded = load_model()
    return {
        "success": True,
        "preferred_mode": preferred_device,
        "active_device": active_device,
        "cuda_available": CUDA_AVAILABLE,
        "model_loaded": model_loaded,
    }


@app.get("/profile")
async def get_profile():
    if PROFILE_FILE.exists():
        try:
            with open(PROFILE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "name": "User",
        "organization": "",
        "language_focus": "gujarati",
        "goal": "Improve Gujarati ASR quality",
        "updated_at": None,
    }


@app.post("/profile")
async def set_profile(request: ProfileRequest):
    payload = {
        "name": request.name.strip() or "User",
        "organization": request.organization.strip(),
        "language_focus": request.language_focus.strip() or "gujarati",
        "goal": request.goal.strip() or "Improve Gujarati ASR quality",
        "updated_at": datetime.now().isoformat(),
    }
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {"success": True, "profile": payload}

@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    """
    Accepts an audio file (wav, mp3, etc.) and returns transcription.
    """
    # Validate file type (optional)
    if not file.filename.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Load and transcribe
        print(f"Transcribing file: {file.filename}", flush=True)
        audio_array, sr = load_audio(tmp_path)
        print(f"Audio loaded: {len(audio_array)} samples at {sr}Hz", flush=True)
        transcription = transcribe_audio(audio_array, sr)
        print(f"Transcription complete: {transcription[:100]}...", flush=True)
        return JSONResponse(content={
            "transcription": transcription,
            "duration": len(audio_array) / sr if sr else 0,
            "confidence": 0.95
        })
    except Exception as e:
        print(f"Transcription error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


@app.post("/subtitle")
async def subtitle_endpoint(file: UploadFile = File(...)):
    """
    Accepts an audio file and returns SRT subtitles.
    """
    if not file.filename.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        audio_array, sr = load_audio(tmp_path)
        segments = transcribe_with_timestamps(audio_array, sr)

        # Convert segments to SRT format
        srt_lines = []
        for i, seg in enumerate(segments, start=1):
            start = seg["start"]
            end = seg["end"]
            text = seg["text"]
            # Format timestamps as HH:MM:SS,mmm
            def format_ts(seconds):
                hrs = int(seconds // 3600)
                mins = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                millis = int((seconds - int(seconds)) * 1000)
                return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"
            srt_lines.append(f"{i}\n{format_ts(start)} --> {format_ts(end)}\n{text}\n")

        srt_content = "\n".join(srt_lines)
        # Return as plain text with .srt extension
        return StreamingResponse(
            iter([srt_content]),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=\"{os.path.splitext(file.filename)[0]}.srt\""}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/sync-timestamps")
async def sync_timestamps_endpoint(
    file: UploadFile = File(...),
    chunk_length_s: float = Form(4.0)
):
    """
    Accepts an audio file and returns editable timestamp segments for sync UI.
    """
    if not file.filename.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    if chunk_length_s < 1.0 or chunk_length_s > 15.0:
        raise HTTPException(status_code=400, detail="chunk_length_s must be between 1 and 15 seconds")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        audio_array, sr = load_audio(tmp_path)
        segments = transcribe_with_timestamps(audio_array, sr, chunk_length_s=chunk_length_s)
        normalized = []
        for idx, seg in enumerate(segments, start=1):
            normalized.append({
                "id": idx,
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": (seg.get("text", "") or "").strip(),
            })
        return {"success": True, "segments": normalized, "count": len(normalized)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/sync-export-srt")
async def sync_export_srt(payload: dict):
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(segments, list) or not segments:
        raise HTTPException(status_code=400, detail="segments list is required")

    def format_ts(seconds):
        s = float(max(0.0, seconds))
        hrs = int(s // 3600)
        mins = int((s % 3600) // 60)
        secs = int(s % 60)
        millis = int((s - int(s)) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    lines = []
    for i, seg in enumerate(segments, start=1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        lines.append(f"{i}\n{format_ts(start)} --> {format_ts(end)}\n{text}\n")
    return {"success": True, "srt": "\n".join(lines), "count": len(lines)}


# ----------------------------
# Stats tracking
# ----------------------------
STATS_FILE = DATA_DIR / "stats.json"

def load_stats():
    if STATS_FILE.exists():
        with open(STATS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"total": 0, "gujarati": 0, "gujlish": 0}

def save_stats(stats):
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False)

def load_manifest_entries():
    manifest_path = DATA_DIR / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries

def append_manifest_entry(entry: dict):
    manifest_path = DATA_DIR / "manifest.jsonl"
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@app.post("/submit-sample")
async def submit_sample(
    transcription: str = Form(""),
    language: str = Form("gujarati"),
    speaker: str = Form(""),
    notes: str = Form(""),
    text_only: str = Form("false"),
    audio: UploadFile = File(None)
):
    """
    Accept audio file with transcription to contribute to training data.
    Can submit text-only samples without audio.
    """
    is_text_only = text_only.lower() == "true" or audio is None
    
    if not transcription.strip():
        raise HTTPException(status_code=400, detail="Transcription is required")
    
    # Generate unique ID
    sample_id = str(uuid.uuid4())[:8]
    audio_path = None
    
    if not is_text_only and audio:
        if not audio.filename.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Save audio file
        audio_ext = os.path.splitext(audio.filename)[1]
        audio_filename = f"{sample_id}{audio_ext}"
        audio_path = DATA_DIR / "wav"
        audio_path.mkdir(parents=True, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_ext) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        
        try:
            import shutil
            shutil.move(tmp_path, audio_path / audio_filename)
            audio_path = str(audio_path / audio_filename)
        except Exception as e:
            os.unlink(tmp_path)
            raise HTTPException(status_code=500, detail=f"Failed to save audio: {e}")
    
    # Save metadata
    metadata = {
        "id": sample_id,
        "audio": audio_path,
        "transcription": transcription.strip(),
        "language": language,
        "speaker": speaker.strip() if speaker else "anonymous",
        "notes": notes.strip(),
        "text_only": is_text_only,
        "created_at": datetime.now().isoformat()
    }
    
    manifest_path = DATA_DIR / "manifest.jsonl"
    with open(manifest_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
    
    # Update stats
    stats = load_stats()
    stats["total"] = stats.get("total", 0) + 1
    if language == "gujarati":
        stats["gujarati"] = stats.get("gujarati", 0) + 1
    else:
        stats["gujlish"] = stats.get("gujlish", 0) + 1
    save_stats(stats)
    
    return JSONResponse(content={
        "success": True,
        "sample_id": sample_id,
        "message": "Sample submitted successfully"
    })


@app.get("/stats")
async def get_stats():
    """Return submission statistics."""
    stats = load_stats()
    return stats


@app.post("/dataset-upload")
async def dataset_upload(
    file: UploadFile = File(...),
    language: str = Form("gujarati")
):
    """
    Import text/audio metadata from CSV or JSONL into submitted_samples manifest.
    Supported keys: transcription/text/sentence, language, speaker, notes, audio(optional path).
    """
    name = (file.filename or "").lower()
    if not (name.endswith(".csv") or name.endswith(".jsonl")):
        raise HTTPException(status_code=400, detail="Only .csv or .jsonl files are supported")

    raw = await file.read()
    try:
        content = raw.decode("utf-8-sig")
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to decode dataset file as UTF-8")

    rows = []
    try:
        if name.endswith(".csv"):
            reader = csv.DictReader(content.splitlines())
            rows = [dict(r) for r in reader]
        else:
            for line in content.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse dataset file: {e}")

    imported = 0
    imported_audio = 0
    imported_text_only = 0
    stats = load_stats()
    for row in rows:
        text = (
            (row.get("transcription") or row.get("text") or row.get("sentence") or "").strip()
            if isinstance(row, dict) else ""
        )
        if not text:
            continue
        row_lang = ((row.get("language") or language or "gujarati") if isinstance(row, dict) else language).strip().lower()
        speaker = ((row.get("speaker") or "dataset_import") if isinstance(row, dict) else "dataset_import").strip()
        notes = ((row.get("notes") or "") if isinstance(row, dict) else "").strip()
        audio_path = ((row.get("audio") or "") if isinstance(row, dict) else "").strip()
        audio_exists = bool(audio_path and Path(audio_path).exists())

        entry = {
            "id": str(uuid.uuid4())[:8],
            "audio": audio_path if audio_exists else None,
            "transcription": text,
            "language": row_lang if row_lang in ("gujarati", "gujlish") else "gujarati",
            "speaker": speaker or "dataset_import",
            "notes": notes,
            "text_only": not audio_exists,
            "created_at": datetime.now().isoformat(),
        }
        append_manifest_entry(entry)
        imported += 1
        if audio_exists:
            imported_audio += 1
        else:
            imported_text_only += 1
        stats["total"] = stats.get("total", 0) + 1
        if entry["language"] == "gujarati":
            stats["gujarati"] = stats.get("gujarati", 0) + 1
        else:
            stats["gujlish"] = stats.get("gujlish", 0) + 1

    save_stats(stats)
    return {
        "success": True,
        "rows_read": len(rows),
        "imported": imported,
        "audio_linked": imported_audio,
        "text_only": imported_text_only,
    }


@app.post("/collect-data")
async def collect_data_from_urls(request: CollectRequest):
    if not request.urls:
        raise HTTPException(status_code=400, detail="At least one URL is required")
    mode = (request.mode or "web_text").strip().lower()
    if mode not in ("web_text", "youtube_audio"):
        raise HTTPException(status_code=400, detail="mode must be web_text or youtube_audio")

    imported = 0
    failed = 0
    errors = []
    stats = load_stats()

    for raw_url in request.urls:
        url = (raw_url or "").strip()
        if not url:
            continue
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            failed += 1
            errors.append({"url": url, "error": "Only http/https URLs are allowed"})
            continue

        try:
            if mode == "web_text":
                try:
                    import requests
                    from bs4 import BeautifulSoup
                except Exception:
                    raise RuntimeError("Install requirements: requests beautifulsoup4")

                resp = requests.get(url, timeout=20, headers={"User-Agent": "GujTranscribeBot/1.0"})
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.extract()
                text = " ".join(soup.get_text(separator=" ").split())
                text = text[:1200].strip()
                if not text:
                    raise RuntimeError("No readable text extracted")

                entry = {
                    "id": str(uuid.uuid4())[:8],
                    "audio": None,
                    "transcription": text,
                    "language": request.language if request.language in ("gujarati", "gujlish") else "gujarati",
                    "speaker": "web_collect",
                    "notes": f"source:{url}",
                    "text_only": True,
                    "created_at": datetime.now().isoformat(),
                }
                append_manifest_entry(entry)
                with open(COLLECTED_DATA_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"type": "web_text", "url": url, "entry_id": entry["id"]}, ensure_ascii=False) + "\n")
                stats["total"] = stats.get("total", 0) + 1
                if entry["language"] == "gujarati":
                    stats["gujarati"] = stats.get("gujarati", 0) + 1
                else:
                    stats["gujlish"] = stats.get("gujlish", 0) + 1
                imported += 1

            else:
                try:
                    import yt_dlp
                except Exception:
                    raise RuntimeError("Install requirement: yt-dlp")

                item_id = str(uuid.uuid4())[:8]
                outtmpl = str(COLLECTED_AUDIO_DIR / f"{item_id}.%(ext)s")
                ydl_opts = {
                    "format": "bestaudio/best",
                    "outtmpl": outtmpl,
                    "quiet": True,
                    "noplaylist": True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    downloaded_path = ydl.prepare_filename(info)
                if not Path(downloaded_path).exists():
                    raise RuntimeError("Audio download failed")
                title = (info.get("title") or "").strip()
                entry = {
                    "id": item_id,
                    "audio": downloaded_path,
                    "transcription": title or "online_collected_audio",
                    "language": request.language if request.language in ("gujarati", "gujlish") else "gujarati",
                    "speaker": "online_collect",
                    "notes": f"source:{url}",
                    "text_only": False,
                    "created_at": datetime.now().isoformat(),
                }
                append_manifest_entry(entry)
                with open(COLLECTED_DATA_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"type": "youtube_audio", "url": url, "audio": downloaded_path, "entry_id": item_id}, ensure_ascii=False) + "\n")
                stats["total"] = stats.get("total", 0) + 1
                if entry["language"] == "gujarati":
                    stats["gujarati"] = stats.get("gujarati", 0) + 1
                else:
                    stats["gujlish"] = stats.get("gujlish", 0) + 1
                imported += 1

        except Exception as e:
            failed += 1
            errors.append({"url": url, "error": str(e)})

    save_stats(stats)
    return {
        "success": True,
        "mode": mode,
        "requested": len(request.urls),
        "imported": imported,
        "failed": failed,
        "errors": errors[:10],
    }


@app.get("/self-improvement-status")
async def self_improvement_status():
    entries = load_manifest_entries()
    audio_entries = [e for e in entries if e.get("audio") and Path(str(e.get("audio"))).exists()]
    text_only_entries = [e for e in entries if not e.get("audio")]
    corrections_count = len(word_corrections)
    training_ready = len(audio_entries) >= 5
    progress = min(100, len(audio_entries) * 12 + corrections_count * 2)
    next_step = "Upload at least 5 valid audio samples to unlock model update." if not training_ready else "Run model update from Sync panel."
    return {
        "total_samples": len(entries),
        "audio_samples": len(audio_entries),
        "text_only_samples": len(text_only_entries),
        "corrections_count": corrections_count,
        "training_ready": training_ready,
        "improvement_progress_percent": progress,
        "next_step": next_step,
        "update_status": model_update_status.get("status", "idle"),
        "update_message": model_update_status.get("message", ""),
    }


@app.post("/correct-word")
async def correct_word(wrong: str = Form(""), correct: str = Form("")):
    """Submit a word correction for self-improvement."""
    if not wrong.strip() or not correct.strip():
        raise HTTPException(status_code=400, detail="Both wrong and correct words required")
    
    save_word_correction(wrong, correct)
    return JSONResponse(content={
        "success": True,
        "message": f"Correction saved: '{wrong}' → '{correct}'",
        "corrections_count": len(word_corrections)
    })


@app.get("/corrections")
async def get_corrections():
    """Get all word corrections."""
    return {"corrections": word_corrections, "count": len(word_corrections)}


@app.post("/clear-corrections")
async def clear_corrections():
    """Clear all word corrections."""
    global word_corrections
    word_corrections = {}
    if WORD_CORRECTIONS_FILE.exists():
        WORD_CORRECTIONS_FILE.unlink()
    return JSONResponse(content={"success": True, "message": "All corrections cleared"})


@app.post("/update-model")
async def update_model():
    """
    Trigger model retraining with submitted samples.
    """
    manifest_path = DATA_DIR / "manifest.jsonl"
    
    if not manifest_path.exists():
        raise HTTPException(status_code=400, detail="No samples submitted yet")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f if line.strip()]

    audio_samples = [s for s in samples if s.get("audio") and Path(str(s.get("audio"))).exists()]

    if len(audio_samples) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 5 valid audio samples. Found {len(audio_samples)} audio / {len(samples)} total."
        )

    with update_lock:
        if model_update_status["status"] == "running":
            raise HTTPException(status_code=409, detail="Model update already running")

        model_update_status.update({
            "status": "running",
            "message": "Preparing training data...",
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "samples_count": len(samples),
            "audio_samples_count": len(audio_samples),
        })

    def _run_update_job(valid_audio_samples):
        global asr_pipeline, model_loaded
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            update_manifest = DATA_DIR / "update_manifest.jsonl"
            with open(update_manifest, "w", encoding="utf-8") as out:
                for s in valid_audio_samples:
                    row = {
                        "audio": str(s["audio"]),
                        "text": (s.get("transcription") or "").strip(),
                    }
                    if row["text"]:
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")

            cmd = [
                sys.executable,
                str(BASE_DIR / "scripts" / "train_whisper.py"),
                "--manifest", str(update_manifest),
                "--output_dir", str(UPDATE_MODEL_DIR),
                "--model_name_or_path", PRETRAINED_MODEL,
                "--language", "gu",
                "--max_steps", os.getenv("UPDATE_MAX_STEPS", "100"),
                "--per_device_train_batch_size", os.getenv("UPDATE_BATCH_SIZE", "2"),
                "--gradient_accumulation_steps", "1",
                "--learning_rate", "1e-5",
                "--warmup_steps", "20",
            ]

            with open(UPDATE_LOG_FILE, "w", encoding="utf-8") as logf:
                logf.write(f"Started at {datetime.now().isoformat()}\n")
                logf.write(" ".join(cmd) + "\n\n")
                logf.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=str(BASE_DIR),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )

            if proc.returncode != 0:
                raise RuntimeError(f"Training failed with exit code {proc.returncode}. Check {UPDATE_LOG_FILE}")

            try:
                asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=str(UPDATE_MODEL_DIR),
                    device=_resolve_pipeline_device()
                )
                model_loaded = True
            except Exception as load_err:
                raise RuntimeError(f"Training finished, but loading updated model failed: {load_err}")

            with update_lock:
                model_update_status.update({
                    "status": "success",
                    "message": f"Model updated successfully using {len(valid_audio_samples)} audio samples.",
                    "finished_at": datetime.now().isoformat(),
                })
        except Exception as e:
            with update_lock:
                model_update_status.update({
                    "status": "error",
                    "message": str(e),
                    "finished_at": datetime.now().isoformat(),
                })

    threading.Thread(target=_run_update_job, args=(audio_samples,), daemon=True).start()

    return JSONResponse(content={
        "success": True,
        "message": f"Model update started with {len(audio_samples)} audio samples.",
        "samples_count": len(samples),
        "audio_samples_count": len(audio_samples),
    })


@app.get("/update-model-status")
async def update_model_status_endpoint():
    status = dict(model_update_status)
    if UPDATE_LOG_FILE.exists():
        try:
            with open(UPDATE_LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            status["log_tail"] = "".join(lines[-40:])
        except Exception:
            status["log_tail"] = ""
    else:
        status["log_tail"] = ""
    return status


@app.get("/health")
async def root():
    return {"message": "Gujarati + Gujlish ASR API is running. Model loaded: " + str(model_loaded)}


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    return {
        "model_loaded": model_loaded,
        "model_name": PRETRAINED_MODEL if model_loaded else "none",
        "preferred_mode": preferred_device,
        "device": active_device,
        "cuda_available": CUDA_AVAILABLE
    }


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming transcription.
    Client sends audio chunks, server returns transcriptions.
    """
    client_id = str(uuid.uuid4())[:8]
    await manager.connect(websocket, client_id)
    print(f"WebSocket client {client_id} connected", flush=True)
    
    audio_buffer = []
    sample_rate = 16000
    
    try:
        await websocket.send_json({
            "type": "connected",
            "client_id": client_id,
            "message": "Ready for audio streaming"
        })
        
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "audio":
                audio_data = data.get("data")
                if audio_data:
                    audio_bytes = base64.b64decode(audio_data)
                    audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                    audio_buffer.append(audio_chunk)
                    
                    if len(audio_buffer) > 0:
                        combined_audio = np.concatenate(audio_buffer)
                        
                        if len(combined_audio) >= sample_rate * 2:
                            try:
                                result = asr_pipeline(combined_audio)
                                text = result.get("text", "")
                                if text.strip():
                                    await websocket.send_json({
                                        "type": "transcription",
                                        "text": text.strip(),
                                        "partial": False
                                    })
                                    audio_buffer = []
                            except Exception as e:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": str(e)
                                })
            
            elif data.get("type") == "finalize":
                if len(audio_buffer) > 0:
                    combined_audio = np.concatenate(audio_buffer)
                    try:
                        result = asr_pipeline(combined_audio)
                        text = result.get("text", "")
                        await websocket.send_json({
                            "type": "transcription",
                            "text": text.strip(),
                            "partial": False
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e)
                        })
                audio_buffer = []
                
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        print(f"WebSocket client {client_id} disconnected", flush=True)
    except Exception as e:
        print(f"WebSocket error: {e}", flush=True)
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        manager.disconnect(client_id)


# ----------------------------
# Run with uvicorn if executed directly
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
