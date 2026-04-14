"""
Gujarati ASR - AI-Powered Speech Recognition
FastAPI Backend with Vocabulary Training
"""
import os
import json
import tempfile
import uuid
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import pipeline
import librosa
import numpy as np
import soundfile as sf

# ============================================
# Configuration
# ============================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WEB_DIR = BASE_DIR / "web"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTIONS_DIR = DATA_DIR / "transcriptions"
TRAINING_DIR = DATA_DIR / "training"

SUPPORTED_EXTENSIONS = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm', '.wma', '.aac')

for d in [DATA_DIR, AUDIO_DIR, TRANSCRIPTIONS_DIR, TRAINING_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================
# Model Management
# ============================================
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = 0 if CUDA_AVAILABLE else -1
DEVICE_NAME = "GPU (CUDA)" if CUDA_AVAILABLE else "CPU"

print(f"{'='*50}")
print(f"Gujarati ASR - Model Management")
print(f"Device: {DEVICE_NAME}")

# Available models
AVAILABLE_MODELS = {
    "gujarati-small": {
        "name": "vasista22/whisper-gujarati-small",
        "display": "Gujarati Small (Recommended)",
        "size": "500MB",
        "description": "Optimized for Gujarati"
    },
    "large-v3": {
        "name": "openai/whisper-large-v3",
        "display": "Whisper Large V3",
        "size": "3GB",
        "description": "Best accuracy, slow"
    },
    "medium": {
        "name": "openai/whisper-medium",
        "display": "Whisper Medium",
        "size": "1.5GB",
        "description": "Good balance"
    },
    "small": {
        "name": "openai/whisper-small",
        "display": "Whisper Small",
        "size": "500MB",
        "description": "Fast, decent accuracy"
    },
    "base": {
        "name": "openai/whisper-base",
        "display": "Whisper Base",
        "size": "150MB",
        "description": "Very fast"
    },
    "tiny": {
        "name": "openai/whisper-tiny",
        "display": "Whisper Tiny",
        "size": "75MB",
        "description": "Fastest, basic"
    }
}

class ModelManager:
    def __init__(self):
        self.loaded_models = {}
        self.current_model_key = None
        self.current_pipeline = None
    
    def load_model(self, model_key: str):
        """Load a model by key"""
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}")
        
        model_info = AVAILABLE_MODELS[model_key]
        model_name = model_info["name"]
        
        # If already loaded, just switch
        if model_key in self.loaded_models:
            self.current_model_key = model_key
            self.current_pipeline = self.loaded_models[model_key]
            print(f"[SWITCH] Using cached: {model_name}", flush=True)
            return model_key
        
        # Load new model
        try:
            print(f"[LOADING] {model_name}...", flush=True)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=DEVICE,
            )
            self.loaded_models[model_key] = pipe
            self.current_model_key = model_key
            self.current_pipeline = pipe
            print(f"[OK] Loaded: {model_name}", flush=True)
            return model_key
        except Exception as e:
            print(f"[FAIL] {model_name}: {e}", flush=True)
            raise e
    
    def get_current_model(self):
        return self.current_model_key
    
    def get_loaded_models(self):
        return list(self.loaded_models.keys())
    
    def unload_model(self, model_key: str):
        """Unload a model to free memory"""
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            if self.current_model_key == model_key:
                self.current_model_key = None
                self.current_pipeline = None
            return True
        return False

model_manager = ModelManager()

# Load default model
try:
    model_manager.load_model("gujarati-small")
except:
    try:
        model_manager.load_model("small")
    except:
        model_manager.load_model("tiny")

# Keep asr_pipeline for backward compatibility
asr_pipeline = model_manager.current_pipeline
LOADED_MODEL = model_manager.current_model_key

# ============================================
# Vocabulary System
# ============================================
VOCABULARY_FILE = DATA_DIR / "vocabulary.json"
CORRECTIONS_FILE = DATA_DIR / "corrections.json"

# Gujarati-to-Gujarati corrections for common model mistakes
GUJARATI_CORRECTIONS = {
    # Names
    "મૂયુર": "મયુર",
    "મૂયુરનું": "મયુરનું",
    "મૂયુરના": "મયુરના",
    "મૂયુરને": "મયુરને",
    "મોહન": "મોહન",
    "રાજેશ": "રાજેશ",
    "પ્રિયા": "પ્રિયા",
    
    # Double matra fixes
    "મારુંં": "મારું",
    "તારુંં": "તારું",
    "સારુંં": "સારું",
    "હુંં": "હું",
    "તુંં": "તું",
    
    # Common word fixes
    "મારુ": "મારું",
    "તારુ": "તારું",
    "સારુ": "સારું",
    "નામું": "નામ",
    "નામે": "નામ",
    "છૂ": "છું",
    "છેં": "છે",
    "છોં": "છો",
    "કેંમ": "કેમ",
    "કેમું": "કેમ",
    "કેં": "કે",
    "તોં": "તો",
    "હોં": "હો",
    "જોં": "જો",
    
    # Place names
    "અમદાવાદું": "અમદાવાદ",
    "અમદાવાદે": "અમદાવાદ",
    "ગુજરાતું": "ગુજરાત",
    "ગુજરાતે": "ગુજરાત",
    "ભારતું": "ભારત",
    "સુરતું": "સુરત",
    "વડોદરાું": "વડોદરા",
    "રાજકોટું": "રાજકોટ",
    "ગાંધીનગરું": "ગાંધીનગર",
    "જામનગરું": "જામનગર",
    "ભાવનગરું": "ભાવનગર",
    "મોરબીું": "મોરબી",
    "મહેસાણાું": "મહેસાણા",
    "પાટણું": "પાટણ",
    "વલ્સાદું": "વલ્સાદ",
    "વાપીું": "વાપી",
    "દ્વારકાું": "દ્વારકા",
    "પોરબંદરું": "પોરબંદર",
    "કચ્છું": "કચ્છ",
    
    # Verbs and particles
    "કરુંં": "કરું",
    "કરેં": "કરે",
    "કરોં": "કરો",
    "જાઉંં": "જાઉં",
    "આવુંં": "આવું",
    "જવુંં": "જવું",
    "થાયું": "થાય",
    "થયું": "થયું",
    
    # Common phrases
    "કેમછો": "કેમ છો",
    "કેમછે": "કેમ છે",
    "હુંછું": "હું છું",
    "તમેછો": "તમે છો",
    "તુંછે": "તું છે",
}

class VocabularyManager:
    def __init__(self):
        self.entries = {}
        self.gujarati_corrections = dict(GUJARATI_CORRECTIONS)
        self.load()

    def load(self):
        if VOCABULARY_FILE.exists():
            with open(VOCABULARY_FILE, "r", encoding="utf-8") as f:
                self.entries = json.load(f)
        else:
            self.entries = {}
        
        # Load custom Gujarati corrections
        if CORRECTIONS_FILE.exists():
            with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
                custom = json.load(f)
                self.gujarati_corrections.update(custom.get("gujarati", {}))

    def save(self):
        with open(VOCABULARY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)
        
        # Save custom Gujarati corrections
        with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump({"gujarati": self.gujarati_corrections}, f, ensure_ascii=False, indent=2)

    def add(self, word: str, gujarati: str, gujlish: str, category: str = "general"):
        entry_id = str(uuid.uuid4())[:8]
        self.entries[entry_id] = {
            "id": entry_id,
            "word": word.lower().strip(),
            "gujarati": gujarati.strip(),
            "gujlish": gujlish.lower().strip(),
            "category": category,
            "timestamp": datetime.now().isoformat(),
        }
        self.save()
        return entry_id

    def add_gujarati_correction(self, wrong: str, correct: str):
        """Add Gujarati-to-Gujarati correction"""
        self.gujarati_corrections[wrong] = correct
        self.save()

    def remove(self, entry_id: str):
        if entry_id in self.entries:
            del self.entries[entry_id]
            self.save()
            return True
        return False

    def get_all(self) -> Dict:
        return self.entries

    def get_by_category(self, category: str) -> List[Dict]:
        return [v for v in self.entries.values() if v.get("category") == category]

    def clear(self):
        self.entries = {}
        self.save()

    def get_corrections_dict(self) -> Dict:
        """Get word -> gujarati mapping for corrections"""
        return {v["word"]: v["gujarati"] for v in self.entries.values() if v.get("word") and v.get("gujarati")}

    def get_gujlish_dict(self) -> Dict:
        """Get gujlish -> gujarati mapping (uses 'word' field which is the Gujlish input)"""
        result = {}
        for v in self.entries.values():
            # Use 'word' field as the Gujlish key (user input)
            if v.get("word") and v.get("gujarati"):
                result[v["word"]] = v["gujarati"]
            # Also add 'gujlish' field if different
            if v.get("gujlish") and v.get("gujarati") and v.get("gujlish") != v.get("word"):
                result[v["gujlish"]] = v["gujarati"]
        return result

    def apply_corrections(self, text: str) -> str:
        """Apply all corrections to transcribed text"""
        # Apply Gujarati-to-Gujarati corrections FIRST (before fixing double matras)
        for wrong, correct in self.gujarati_corrections.items():
            if wrong in text:
                text = text.replace(wrong, correct)
        
        # Then fix remaining double matras
        while '\u0a82\u0a82' in text:
            text = text.replace('\u0a82\u0a82', '\u0a82')
        
        # Apply vocabulary corrections
        corrections = self.get_corrections_dict()
        words = text.split()
        corrected = []
        for word in words:
            clean = word.strip(".,!?")
            if clean.lower() in corrections:
                corrected.append(corrections[clean.lower()])
            else:
                corrected.append(word)
        return " ".join(corrected)

    def search(self, query: str) -> List[Dict]:
        """Search vocabulary entries"""
        query = query.lower()
        return [v for v in self.entries.values()
                if query in v.get("word", "").lower() or
                   query in v.get("gujarati", "").lower() or
                   query in v.get("gujlish", "").lower()]

    def get_stats(self) -> Dict:
        categories = {}
        for v in self.entries.values():
            cat = v.get("category", "general")
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total": len(self.entries),
            "categories": categories,
            "gujarati_corrections": len(self.gujarati_corrections),
        }

vocab = VocabularyManager()

# ============================================
# Transliteration
# ============================================

# Word-level dictionary for natural Gujlish
GUJARATI_TO_GUJLISH_WORDS = {
    'મારું': 'maru',
    'તારું': 'taru',
    'તમારું': 'tamaru',
    'સારું': 'saaru',
    'નામ': 'naam',
    'મયુર': 'mayur',
    'હું': 'hu',
    'છું': 'chhu',
    'છે': 'chhe',
    'છો': 'chho',
    'કેમ': 'kem',
    'શું': 'shu',
    'તમે': 'tame',
    'તું': 'tu',
    'એક': 'ek',
    'બે': 'be',
    'ત્રણ': 'tran',
    'ચાર': 'char',
    'પાંચ': 'paanch',
    'સારો': 'saaro',
    'સારી': 'saari',
    'મોટો': 'moto',
    'મોટી': 'moti',
    'નાનો': 'nano',
    'નાની': 'nani',
    'ઘર': 'ghar',
    'પાણી': 'paani',
    'ખાવું': 'khavu',
    'પીવું': 'pivu',
    'જવું': 'javu',
    'આવવું': 'aavvu',
    'કરવું': 'karvu',
    'બોલવું': 'bolvu',
    'જોવું': 'jovu',
    'સાંભળવું': 'sambhalvu',
    'વાંચવું': 'vanchvu',
    'લખવું': 'lakhvu',
    'ગુજરાત': 'gujarat',
    'ગુજરાતી': 'gujarati',
    'ભારત': 'bharat',
    'અમદાવાદ': 'ahmedabad',
    'વડોદરા': 'vadodara',
    'સુરત': 'surat',
    'રાજકોટ': 'rajkot',
    'ગાંધીનગર': 'gandhinagar',
    'દેશ': 'desh',
    'શહેર': 'sheher',
    'ગામ': 'gaam',
    'ભાઈ': 'bhai',
    'બહેન': 'bahen',
    'મા': 'maa',
    'બાપુજી': 'bapuji',
    'પપ્પા': 'pappa',
    'મમ્મી': 'mummy',
    'દાદા': 'dada',
    'દાદી': 'dadi',
    'નાના': 'nana',
    'નાની': 'nani',
    'છોકરો': 'chokro',
    'છોકરી': 'chokri',
    'માણસ': 'maanas',
    'લોકો': 'loko',
    'દોસ્ત': 'dost',
    'મિત્ર': 'mitra',
    'પ્રેમ': 'prem',
    'ખુશી': 'khushi',
    'દુઃખ': 'dukh',
    'ગમે': 'game',
    'નથી': 'nathi',
    'હા': 'haa',
    'ના': 'na',
    'બરાબર': 'barabar',
    'સાચું': 'sachu',
    'ખોટું': 'khotu',
    'હવે': 'have',
    'પછી': 'pachi',
    'પહેલાં': 'pahela',
    'આજે': 'aaje',
    'કાલે': 'kaale',
    'ગઈકાલે': 'gikale',
    'સવારે': 'savaare',
    'બપોરે': 'bapore',
    'સાંજે': 'saanje',
    'રાત્રે': 'raatre',
    'દિવસ': 'divas',
    'રાત': 'raat',
    'સમય': 'samay',
    'વર્ષ': 'varsh',
    'મહિનો': 'mahino',
    'અઠવાડિયું': 'athvadiyu',
    'આ': 'aa',
    'તે': 'te',
    'એ': 'e',
    'ત્યાં': 'tya',
    'અહીં': 'ahi',
    'ક્યાં': 'kya',
    'ક્યારે': 'kyare',
    'કેટલું': 'ketlu',
    'કેમ': 'kem',
    'શા': 'sha',
    'માટે': 'mate',
    'માં': 'ma',
    'થી': 'thi',
    'ને': 'ne',
    'ના': 'na',
    'નું': 'nu',
    'નો': 'no',
    'ની': 'ni',
    'માં': 'ma',
    'પર': 'par',
    'સાથે': 'sathe',
    'વિશે': 'vishe',
    'જેવું': 'jevu',
    'તેવું': 'tevu',
}

def gujarati_to_gujlish(text: str) -> str:
    """Convert Gujarati script to Gujlish (romanized)"""
    # Character-level mapping for fallback
    char_map = {
        'અ': 'a', 'આ': 'aa', 'ઇ': 'i', 'ઈ': 'ee', 'ઉ': 'u', 'ઊ': 'oo', 'ઋ': 'ri',
        'એ': 'e', 'ઐ': 'ai', 'ઓ': 'o', 'ઔ': 'au',
        'ક': 'k', 'ખ': 'kh', 'ગ': 'g', 'ઘ': 'gh', 'ઙ': 'ng',
        'ચ': 'ch', 'છ': 'chh', 'જ': 'j', 'ઝ': 'jh', 'ઞ': 'ny',
        'ત': 't', 'થ': 'th', 'દ': 'd', 'ધ': 'dh', 'ન': 'n',
        'પ': 'p', 'ફ': 'ph', 'બ': 'b', 'ભ': 'bh', 'મ': 'm',
        'ય': 'y', 'ર': 'r', 'લ': 'l', 'વ': 'v', 'શ': 'sh', 'ષ': 'sh', 'સ': 's', 'હ': 'h',
        'ળ': 'l', 'ક્ષ': 'ksh', 'જ્ઞ': 'gy',
        '્': '', 'ા': 'a', 'િ': 'i', 'ી': 'ee', 'ુ': 'u', 'ૂ': 'oo',
        'ે': 'e', 'ૈ': 'ai', 'ો': 'o', 'ૌ': 'au',
        'ં': '', 'ઃ': 'h',
    }
    
    # First try word-level replacement
    result = text
    for guj, gujlish in sorted(GUJARATI_TO_GUJLISH_WORDS.items(), key=lambda x: -len(x[0])):
        if guj in result:
            result = result.replace(guj, gujlish)
    
    # If still has Gujarati characters, do char-level conversion
    if any('\u0a80' <= c <= '\u0aff' for c in result):
        converted = []
        i = 0
        while i < len(result):
            c = result[i]
            if '\u0a80' <= c <= '\u0aff':
                # It's a Gujarati character
                if c in char_map:
                    converted.append(char_map[c])
                else:
                    converted.append(c)
            else:
                converted.append(c)
            i += 1
        result = ''.join(converted)
    
    # Clean up
    result = ' '.join(result.split())  # normalize spaces
    return result.strip()


def gujlish_to_gujarati(text: str) -> str:
    word_dict = {
        'hu': 'હું', 'chu': 'છું', 'chhu': 'છું', 'che': 'છે', 'chhe': 'છે',
        'cho': 'છો', 'chho': 'છો', 'tame': 'તમે', 'tamaru': 'તમારું',
        'maru': 'મારું', 'mane': 'મને', 'kem': 'કેમ', 'shu': 'શું', 'su': 'શું',
        'ek': 'એક', 'naam': 'નામ', 'mayur': 'મયુર', 'saaru': 'સારું',
        'gujarat': 'ગુજરાત', 'gujrati': 'ગુજરાતી', 'india': 'ભારત',
        'bharat': 'ભારત', 'to': 'તો', 'mate': 'માટે', 'ne': 'ને',
        'na': 'ના', 'no': 'નો', 'nu': 'નું', 'ke': 'કે', 'aa': 'આ',
        'je': 'જે', 'te': 'તે', 'badhu': 'બધું', 'koi': 'કોઈ',
        'bhai': 'ભાઈ', 'desh': 'દેશ', 'raaj': 'રાજ',
    }
    
    # Add vocabulary entries
    vocab_dict = vocab.get_gujlish_dict()
    word_dict.update(vocab_dict)
    
    consonants = {
        'k': 'ક', 'kh': 'ખ', 'g': 'ગ', 'gh': 'ઘ',
        'ch': 'ચ', 'chh': 'છ', 'j': 'જ', 'jh': 'ઝ',
        't': 'ત', 'th': 'થ', 'd': 'દ', 'dh': 'ધ', 'n': 'ન',
        'p': 'પ', 'ph': 'ફ', 'f': 'ફ', 'b': 'બ', 'bh': 'ભ', 'm': 'મ',
        'y': 'ય', 'r': 'ર', 'l': 'લ', 'v': 'વ', 'w': 'વ',
        'sh': 'શ', 's': 'સ', 'h': 'હ',
    }
    vowels = {
        'aa': 'આ', 'ee': 'ઈ', 'oo': 'ઊ', 'ii': 'ઈ', 'uu': 'ઊ',
        'ai': 'ઐ', 'au': 'ઔ',
        'a': 'અ', 'i': 'ઇ', 'u': 'ઉ', 'e': 'એ', 'o': 'ઓ',
    }
    matras = {
        'aa': 'ા', 'ee': 'ી', 'oo': 'ૂ', 'ii': 'ી', 'uu': 'ૂ',
        'ai': 'ૈ', 'au': 'ૌ',
        'a': 'ા', 'i': 'િ', 'u': 'ુ', 'e': 'ે', 'o': 'ો',
    }

    words = text.split()
    result = []
    for word in words:
        lower = word.lower()
        if lower in word_dict:
            result.append(word_dict[lower])
            continue

        converted = _phonetic_convert(lower, consonants, vowels, matras)
        result.append(converted if converted else word)
    return ' '.join(result)


def _phonetic_convert(word, consonants, vowels, matras):
    if not word:
        return ''
    result = []
    i = 0
    clusters = ['chh', 'kh', 'gh', 'dh', 'th', 'ph', 'bh', 'jh', 'sh', 'ng', 'ny']

    while i < len(word):
        matched = False
        for cluster in clusters:
            if word[i:].startswith(cluster):
                result.append(consonants.get(cluster, cluster))
                i += len(cluster)
                if i < len(word):
                    v = _get_vowel(word[i:])
                    if v:
                        result.append(matras.get(v, v))
                        i += len(v)
                matched = True
                break
        if matched:
            continue

        c = word[i]
        if c in consonants:
            result.append(consonants[c])
            i += 1
            if i < len(word):
                v = _get_vowel(word[i:])
                if v:
                    result.append(matras.get(v, v))
                    i += len(v)
        elif c in vowels:
            result.append(vowels[c])
            i += 1
        else:
            result.append(c)
            i += 1
    return ''.join(result)


def _get_vowel(text):
    for pattern in ['aa', 'ee', 'oo', 'ii', 'uu', 'ai', 'au', 'a', 'i', 'u', 'e', 'o']:
        if text.startswith(pattern):
            return pattern
    return None

# ============================================
# Audio Processing
# ============================================
def load_audio(file_path: str, target_sr: int = 16000):
    import subprocess
    
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return np.asarray(audio, dtype=np.float32), target_sr
    except Exception:
        pass

    try:
        audio, sr = sf.read(file_path, always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio, target_sr
    except Exception:
        pass

    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', file_path, '-ar', str(target_sr), '-ac', '1', '-f', 'wav', tmp_wav],
            capture_output=True, check=True
        )
        audio, sr = sf.read(tmp_wav, always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, target_sr
    except Exception as e:
        raise ValueError(f"Cannot read audio file: {e}")
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

# ============================================
# FastAPI App
# ============================================
app = FastAPI(title="Gujarati ASR", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ============================================
# Models
# ============================================
class TranslateRequest(BaseModel):
    text: str
    source: str = "gujlish"
    target: str = "gujarati"


class CorrectionRequest(BaseModel):
    wrong: str
    correct: str


class VocabularyRequest(BaseModel):
    word: str
    gujarati: str
    gujlish: str = ""
    category: str = "general"


class VocabularyUpdateRequest(BaseModel):
    id: str
    word: str = ""
    gujarati: str = ""
    gujlish: str = ""
    category: str = ""


class TrainingSampleRequest(BaseModel):
    transcription: str
    gujlish: str = ""
    category: str = "general"


# ============================================
# API Endpoints
# ============================================
@app.get("/")
async def root():
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Gujarati ASR API"}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": LOADED_MODEL is not None,
        "model_name": LOADED_MODEL or "none",
        "current_model": model_manager.get_current_model(),
        "loaded_models": model_manager.get_loaded_models(),
        "device": DEVICE_NAME,
        "cuda": CUDA_AVAILABLE,
        "vocabulary": vocab.get_stats(),
    }


# ============================================
# Model Endpoints
# ============================================
@app.get("/api/models")
async def get_models():
    """Get available models"""
    models = []
    for key, info in AVAILABLE_MODELS.items():
        models.append({
            "key": key,
            "name": info["name"],
            "display": info["display"],
            "size": info["size"],
            "description": info["description"],
            "loaded": key in model_manager.get_loaded_models(),
            "current": key == model_manager.get_current_model()
        })
    return {"models": models, "current": model_manager.get_current_model()}


@app.post("/api/models/switch")
async def switch_model(request: dict):
    """Switch to a different model"""
    global asr_pipeline, LOADED_MODEL
    model_key = request.get("model")
    
    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(400, f"Unknown model: {model_key}")
    
    try:
        model_manager.load_model(model_key)
        asr_pipeline = model_manager.current_pipeline
        LOADED_MODEL = model_manager.get_current_model()
        
        return {
            "success": True,
            "model": model_key,
            "display": AVAILABLE_MODELS[model_key]["display"]
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {str(e)}")


@app.delete("/api/models/{model_key}")
async def unload_model(model_key: str):
    """Unload a model to free memory"""
    if model_manager.unload_model(model_key):
        return {"success": True, "unloaded": model_key}
    raise HTTPException(404, "Model not loaded")


@app.get("/api/test-correction")
async def test_correction():
    """Test endpoint to verify corrections work"""
    test_text = "મારુંં નામ મયુર છે"
    corrected = vocab.apply_corrections(test_text)
    return {
        "original": test_text,
        "corrected": corrected,
        "changed": test_text != corrected,
    }


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(400, f"Unsupported format. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    ext = os.path.splitext(file.filename)[1]
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        print(f"Transcribing: {file.filename}", flush=True)

        current_pipeline = model_manager.current_pipeline
        if current_pipeline is None:
            raise HTTPException(500, "Model not loaded")

        audio, sr = load_audio(tmp_path)
        duration = len(audio) / sr

        result = current_pipeline(audio, chunk_length_s=30, stride_length_s=5)
        text = result.get("text", "").strip()

        # Apply vocabulary corrections
        text = vocab.apply_corrections(text)

        word_count = len(text.split()) if text else 0

        history_entry = {
            "id": str(uuid.uuid4())[:8],
            "filename": file.filename,
            "transcription": text,
            "gujlish": gujarati_to_gujlish(text) if text else "",
            "duration": round(duration, 2),
            "words": word_count,
            "timestamp": datetime.now().isoformat(),
        }
        _save_history(history_entry)

        print(f"Result: {text[:100]}...", flush=True)

        return {
            "success": True,
            "transcription": text,
            "gujlish": history_entry["gujlish"],
            "duration": duration,
            "words": word_count,
            "confidence": 0.95,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}", flush=True)
        raise HTTPException(500, str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/translate")
async def translate(request: TranslateRequest):
    text = request.text.strip()
    if not text:
        return {"translation": ""}

    try:
        if request.source == "gujlish" and request.target == "gujarati":
            translation = gujlish_to_gujarati(text)
        elif request.source == "gujarati" and request.target == "gujlish":
            translation = gujarati_to_gujlish(text)
        else:
            translation = text
        return {"translation": translation}
    except Exception as e:
        raise HTTPException(500, str(e))


# ============================================
# History
# ============================================
@app.get("/api/history")
async def get_history():
    history_file = DATA_DIR / "history.json"
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"items": [], "total": 0}


@app.delete("/api/history")
async def clear_history():
    history_file = DATA_DIR / "history.json"
    if history_file.exists():
        history_file.unlink()
    return {"success": True}


# ============================================
# Stats
# ============================================
@app.get("/api/stats")
async def get_stats():
    history_file = DATA_DIR / "history.json"
    total = 0
    total_duration = 0
    total_words = 0
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            items = data.get("items", [])
            total = len(items)
            total_duration = sum(i.get("duration", 0) for i in items)
            total_words = sum(i.get("words", 0) for i in items)
    
    vocab_stats = vocab.get_stats()
    
    return {
        "total_transcriptions": total,
        "total_duration": round(total_duration, 1),
        "total_words": total_words,
        "model": LOADED_MODEL or "none",
        "device": DEVICE_NAME,
        "vocabulary": vocab_stats["total"],
        "training_samples": len(list(TRAINING_DIR.glob("*.json"))),
    }


# ============================================
# Batch Transcription
# ============================================
@app.post("/api/transcribe/batch")
async def transcribe_batch(files: List[UploadFile] = File(...)):
    """Transcribe multiple audio files"""
    results = []
    
    for file in files:
        if not file.filename.lower().endswith(SUPPORTED_EXTENSIONS):
            results.append({
                "filename": file.filename,
                "success": False,
                "error": f"Unsupported format"
            })
            continue
        
        ext = os.path.splitext(file.filename)[1]
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            current_pipeline = model_manager.current_pipeline
            if current_pipeline is None:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Model not loaded"
                })
                continue

            audio, sr = load_audio(tmp_path)
            duration = len(audio) / sr

            result = current_pipeline(audio, chunk_length_s=30, stride_length_s=5)
            text = result.get("text", "").strip()
            text = vocab.apply_corrections(text)

            # Save to history
            history_entry = {
                "id": str(uuid.uuid4())[:8],
                "filename": file.filename,
                "transcription": text,
                "gujlish": gujarati_to_gujlish(text) if text else "",
                "duration": round(duration, 2),
                "words": len(text.split()) if text else 0,
                "timestamp": datetime.now().isoformat(),
            }
            _save_history(history_entry)

            results.append({
                "filename": file.filename,
                "success": True,
                "transcription": text,
                "gujlish": history_entry["gujlish"],
                "duration": duration,
                "words": history_entry["words"]
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return {"results": results, "total": len(files), "success": sum(1 for r in results if r.get("success"))}


# ============================================
# Word Frequency Analysis
# ============================================
@app.get("/api/analysis/word-frequency")
async def word_frequency():
    """Get word frequency from all transcriptions"""
    history_file = DATA_DIR / "history.json"
    if not history_file.exists():
        return {"words": [], "total": 0}
    
    with open(history_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    word_count = {}
    for item in data.get("items", []):
        text = item.get("transcription", "")
        if text:
            words = text.split()
            for word in words:
                word = word.strip(".,!?")
                if word and len(word) > 1:
                    word_count[word] = word_count.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "words": [{"word": w, "count": c} for w, c in sorted_words[:50]],
        "total_unique": len(word_count),
        "total_words": sum(word_count.values())
    }


# ============================================
# Export Endpoints
# ============================================
@app.get("/api/history/export/txt")
async def export_history_txt():
    """Export history as TXT"""
    history_file = DATA_DIR / "history.json"
    if not history_file.exists():
        raise HTTPException(404, "No history")
    
    with open(history_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    txt_content = "Gujarati ASR - Transcription History\n"
    txt_content += "=" * 50 + "\n\n"
    
    for item in data.get("items", []):
        txt_content += f"File: {item.get('filename', 'N/A')}\n"
        txt_content += f"Date: {item.get('timestamp', 'N/A')}\n"
        txt_content += f"Duration: {item.get('duration', 0)}s\n"
        txt_content += f"Transcription:\n{item.get('transcription', '')}\n"
        txt_content += "-" * 40 + "\n\n"
    
    # Save to temp file
    txt_path = DATA_DIR / "export.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)
    
    return FileResponse(
        path=str(txt_path),
        filename="gujarati_asr_export.txt",
        media_type="text/plain"
    )


@app.get("/api/history/search")
async def search_history(q: str = ""):
    """Search through transcription history"""
    if not q:
        return {"items": [], "total": 0}
    
    history_file = DATA_DIR / "history.json"
    if not history_file.exists():
        return {"items": [], "total": 0}
    
    with open(history_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    q_lower = q.lower()
    results = []
    for item in data.get("items", []):
        text = item.get("transcription", "").lower()
        filename = item.get("filename", "").lower()
        if q_lower in text or q_lower in filename:
            results.append(item)
    
    return {"items": results, "total": len(results), "query": q}


# ============================================
# Corrections (Gujarati-to-Gujarati)
# ============================================

@app.post("/api/correct")
async def add_correction(request: CorrectionRequest):
    """Add Gujarati-to-Gujarati correction for fixing model mistakes"""
    vocab.add_gujarati_correction(request.wrong.strip(), request.correct.strip())
    return {"success": True, "count": len(vocab.gujarati_corrections)}


@app.get("/api/corrections")
async def get_corrections():
    """Get all Gujarati-to-Gujarati corrections"""
    return vocab.gujarati_corrections


@app.delete("/api/corrections")
async def clear_corrections():
    """Clear all corrections"""
    vocab.gujarati_corrections = dict(GUJARATI_CORRECTIONS)  # Reset to defaults
    vocab.save()
    return {"success": True}


# ============================================
# Vocabulary Endpoints
# ============================================
@app.post("/api/vocabulary")
async def add_vocabulary(request: VocabularyRequest):
    entry_id = vocab.add(
        word=request.word,
        gujarati=request.gujarati,
        gujlish=request.gujlish or gujarati_to_gujlish(request.gujarati),
        category=request.category,
    )
    return {"success": True, "id": entry_id, "stats": vocab.get_stats()}


@app.get("/api/vocabulary")
async def get_vocabulary(category: Optional[str] = None, search: Optional[str] = None):
    if search:
        return {"entries": vocab.search(search), "stats": vocab.get_stats()}
    if category:
        return {"entries": vocab.get_by_category(category), "stats": vocab.get_stats()}
    return {"entries": list(vocab.get_all().values()), "stats": vocab.get_stats()}


@app.delete("/api/vocabulary/{entry_id}")
async def delete_vocabulary(entry_id: str):
    if vocab.remove(entry_id):
        return {"success": True, "stats": vocab.get_stats()}
    raise HTTPException(404, "Entry not found")


@app.delete("/api/vocabulary")
async def clear_vocabulary():
    vocab.clear()
    return {"success": True, "stats": vocab.get_stats()}


@app.post("/api/vocabulary/import")
async def import_vocabulary(file: UploadFile = File(...)):
    try:
        content = await file.read()
        data = json.loads(content)
        
        if isinstance(data, list):
            count = 0
            for item in data:
                if "word" in item and "gujarati" in item:
                    vocab.add(
                        word=item["word"],
                        gujarati=item["gujarati"],
                        gujlish=item.get("gujlish", ""),
                        category=item.get("category", "imported"),
                    )
                    count += 1
            return {"success": True, "imported": count, "stats": vocab.get_stats()}
        
        raise HTTPException(400, "Invalid format: expected array of {word, gujarati}")
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON file")


@app.get("/api/vocabulary/export")
async def export_vocabulary():
    entries = list(vocab.get_all().values())
    return JSONResponse(
        content=entries,
        headers={"Content-Disposition": "attachment; filename=vocabulary.json"}
    )


# ============================================
# Training Endpoints
# ============================================
@app.post("/api/training/sample")
async def add_training_sample(
    file: UploadFile = File(None),
    transcription: str = Form(...),
    gujlish: str = Form(""),
    category: str = Form("general"),
):
    sample_id = str(uuid.uuid4())[:8]
    sample = {
        "id": sample_id,
        "transcription": transcription.strip(),
        "gujlish": gujlish.strip() or gujarati_to_gujlish(transcription),
        "category": category,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save audio if provided
    if file and file.filename:
        ext = os.path.splitext(file.filename)[1]
        audio_path = TRAINING_DIR / f"{sample_id}{ext}"
        with open(audio_path, "wb") as f:
            f.write(await file.read())
        sample["audio_file"] = audio_path.name
    
    # Save sample metadata
    sample_path = TRAINING_DIR / f"{sample_id}.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    
    return {"success": True, "id": sample_id}


@app.get("/api/training/samples")
async def get_training_samples():
    samples = []
    for p in TRAINING_DIR.glob("*.json"):
        with open(p, "r", encoding="utf-8") as f:
            samples.append(json.load(f))
    return {"samples": sorted(samples, key=lambda x: x.get("timestamp", ""), reverse=True)}


@app.delete("/api/training/samples/{sample_id}")
async def delete_training_sample(sample_id: str):
    sample_path = TRAINING_DIR / f"{sample_id}.json"
    if sample_path.exists():
        sample_path.unlink()
        for ext in SUPPORTED_EXTENSIONS:
            audio_path = TRAINING_DIR / f"{sample_id}{ext}"
            if audio_path.exists():
                audio_path.unlink()
        return {"success": True}
    raise HTTPException(404, "Sample not found")


@app.delete("/api/training/samples")
async def clear_training_samples():
    for p in TRAINING_DIR.glob("*"):
        p.unlink()
    return {"success": True}


@app.post("/api/training/process")
async def process_training():
    """Process training samples - add vocabulary entries from training data"""
    samples = []
    for p in TRAINING_DIR.glob("*.json"):
        with open(p, "r", encoding="utf-8") as f:
            samples.append(json.load(f))
    
    added = 0
    for sample in samples:
        text = sample.get("transcription", "")
        if text:
            words = text.split()
            for word in words:
                if len(word) > 1:
                    existing = vocab.search(word)
                    if not existing:
                        gujlish = sample.get("gujlish", "")
                        vocab.add(
                            word=word,
                            gujarati=word,
                            gujlish=gujlish,
                            category="training",
                        )
                        added += 1
    
    return {
        "success": True,
        "processed": len(samples),
        "added_to_vocab": added,
        "stats": vocab.get_stats(),
    }


# ============================================
# Subtitle Endpoints
# ============================================
SUBTITLE_DIR = DATA_DIR / "subtitles"
SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)


class SubtitleSegment(BaseModel):
    index: int
    start: float
    end: float
    text: str


class SubtitleUpdateRequest(BaseModel):
    segments: list[SubtitleSegment]


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(segments: list) -> str:
    """Generate SRT content from segments"""
    srt_content = []
    for i, seg in enumerate(segments, 1):
        start = format_srt_time(seg["start"])
        end = format_srt_time(seg["end"])
        text = seg["text"].strip()
        srt_content.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(srt_content)


@app.post("/api/subtitle/transcribe")
async def subtitle_transcribe(file: UploadFile = File(...), chunk_duration: int = Form(3)):
    """Transcribe audio with timestamps for subtitles"""
    if not file.filename.lower().endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(400, f"Unsupported format")

    ext = os.path.splitext(file.filename)[1]
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        print(f"Subtitle transcribing: {file.filename}", flush=True)

        if asr_pipeline is None:
            raise HTTPException(500, "Model not loaded")

        audio, sr = load_audio(tmp_path)
        duration = len(audio) / sr

        # Get transcription (without timestamps to avoid config issues)
        result = asr_pipeline(
            audio,
            chunk_length_s=30,
            stride_length_s=5
        )

        text = result.get("text", "").strip()
        text = vocab.apply_corrections(text)
        
        # Create segments by splitting text based on duration
        segments = []
        if text:
            words = text.split()
            # Calculate words per segment based on chunk_duration
            # Average speaking rate is ~150 words/min = ~2.5 words/sec
            words_per_chunk = max(1, int(chunk_duration * 2.5))
            
            for i in range(0, len(words), words_per_chunk):
                chunk_words = words[i:i + words_per_chunk]
                start_time = round((i / len(words)) * duration, 3)
                end_time = round(min(((i + words_per_chunk) / len(words)) * duration, duration), 3)
                
                segments.append({
                    "index": len(segments) + 1,
                    "start": start_time,
                    "end": end_time,
                    "text": " ".join(chunk_words)
                })

        # Save subtitle data
        subtitle_id = str(uuid.uuid4())[:8]
        subtitle_data = {
            "id": subtitle_id,
            "filename": file.filename,
            "duration": round(duration, 2),
            "segments": segments,
            "timestamp": datetime.now().isoformat()
        }
        
        subtitle_path = SUBTITLE_DIR / f"{subtitle_id}.json"
        with open(subtitle_path, "w", encoding="utf-8") as f:
            json.dump(subtitle_data, f, ensure_ascii=False, indent=2)

        print(f"Generated {len(segments)} segments", flush=True)

        return {
            "success": True,
            "id": subtitle_id,
            "filename": file.filename,
            "duration": duration,
            "segments": segments,
            "segment_count": len(segments)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Subtitle error: {e}", flush=True)
        raise HTTPException(500, str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/api/subtitle/{subtitle_id}")
async def get_subtitle(subtitle_id: str):
    """Get subtitle data by ID"""
    subtitle_path = SUBTITLE_DIR / f"{subtitle_id}.json"
    if not subtitle_path.exists():
        raise HTTPException(404, "Subtitle not found")
    
    with open(subtitle_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.put("/api/subtitle/{subtitle_id}")
async def update_subtitle(subtitle_id: str, request: SubtitleUpdateRequest):
    """Update subtitle segments"""
    subtitle_path = SUBTITLE_DIR / f"{subtitle_id}.json"
    if not subtitle_path.exists():
        raise HTTPException(404, "Subtitle not found")
    
    with open(subtitle_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Update segments
    data["segments"] = [seg.dict() for seg in request.segments]
    data["updated_at"] = datetime.now().isoformat()
    
    with open(subtitle_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return {"success": True, "segments": data["segments"]}


@app.get("/api/subtitle/{subtitle_id}/srt")
async def download_srt(subtitle_id: str):
    """Download subtitle as SRT file"""
    subtitle_path = SUBTITLE_DIR / f"{subtitle_id}.json"
    if not subtitle_path.exists():
        raise HTTPException(404, "Subtitle not found")
    
    with open(subtitle_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    srt_content = generate_srt(data["segments"])
    
    # Save SRT file
    srt_path = SUBTITLE_DIR / f"{subtitle_id}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    
    return FileResponse(
        path=str(srt_path),
        filename=f"{data.get('filename', 'subtitle')}.srt",
        media_type="text/plain"
    )


@app.get("/api/subtitle/{subtitle_id}/srt-content")
async def get_srt_content(subtitle_id: str):
    """Get SRT content as text"""
    subtitle_path = SUBTITLE_DIR / f"{subtitle_id}.json"
    if not subtitle_path.exists():
        raise HTTPException(404, "Subtitle not found")
    
    with open(subtitle_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    srt_content = generate_srt(data["segments"])
    return {"srt": srt_content, "filename": data.get("filename", "subtitle")}


@app.get("/api/subtitles")
async def list_subtitles():
    """List all subtitles"""
    subtitles = []
    for p in SUBTITLE_DIR.glob("*.json"):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            subtitles.append({
                "id": data.get("id"),
                "filename": data.get("filename"),
                "duration": data.get("duration"),
                "segment_count": len(data.get("segments", [])),
                "timestamp": data.get("timestamp")
            })
    return {"subtitles": sorted(subtitles, key=lambda x: x.get("timestamp", ""), reverse=True)}


@app.delete("/api/subtitle/{subtitle_id}")
async def delete_subtitle(subtitle_id: str):
    """Delete subtitle"""
    subtitle_path = SUBTITLE_DIR / f"{subtitle_id}.json"
    srt_path = SUBTITLE_DIR / f"{subtitle_id}.srt"
    
    if subtitle_path.exists():
        subtitle_path.unlink()
    if srt_path.exists():
        srt_path.unlink()
    
    return {"success": True}


# ============================================
# History Helpers
# ============================================
# History Helpers
# ============================================
def _save_history(entry):
    history_file = DATA_DIR / "history.json"
    data = {"items": [], "total": 0}
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    data["items"].insert(0, entry)
    data["total"] = len(data["items"])
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================
# Run
# ============================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
