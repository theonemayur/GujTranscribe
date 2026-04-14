# GujTranscribe - Gujarati Speech Recognition

AI-powered Gujarati speech-to-text web application with multiple models, subtitle generation, and vocabulary management.

## Features

### Core Features
- **Speech Recognition** - Convert Gujarati audio to text
- **Multiple AI Models** - 6 models from tiny to large-v3
- **Voice Recording** - Record directly from browser microphone
- **Batch Processing** - Transcribe multiple files at once
- **Subtitles** - Generate and edit SRT files with timestamps
- **Text Converter** - Convert between Gujlish and Gujarati script

### Advanced Features
- **Vocabulary Management** - Add custom words for better accuracy
- **Word Corrections** - Fix common transcription errors
- **History & Search** - Browse and search past transcriptions
- **Word Frequency Analysis** - See most used words
- **Export Options** - Download as TXT or SRT
- **Dark/Light Theme** - Toggle between themes
- **Keyboard Shortcuts** - Quick actions with hotkeys

## Available Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| whisper-tiny | 75MB | Fastest | Basic |
| whisper-base | 150MB | Very Fast | Good |
| whisper-small | 500MB | Fast | Better |
| whisper-medium | 1.5GB | Medium | Great |
| whisper-large-v3 | 3GB | Slow | Best |
| whisper-gujarati-small | 500MB | Fast | Gujarati Optimized |

## Installation

### Prerequisites
- Python 3.10+
- FFmpeg (for audio processing)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/GujTranscribe.git
cd GujTranscribe

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the application
cd gujarati_asr
python main.py
```

### Access

Open browser and go to: http://localhost:8000

## Usage

### Transcribe Audio
1. Go to **Transcribe** page
2. Upload audio file (WAV, MP3, OGG, FLAC)
3. Click **Transcribe**
4. Copy or download result

### Generate Subtitles
1. Go to **Subtitles** page
2. Upload audio file
3. Click **Generate**
4. Edit segments as needed
5. Download SRT file

### Voice Recording
1. Go to **Record** page
2. Click microphone to start
3. Speak in Gujarati
4. Click stop when done
5. Click **Transcribe**

### Switch Models
1. Go to **Models** page
2. Click on any model to switch
3. First use will download the model

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+U | Upload file |
| Ctrl+T | Transcribe |
| Ctrl+C | Copy result |
| Ctrl+D | Download |

## API Endpoints

### Transcription
- `POST /api/transcribe` - Transcribe single audio file
- `POST /api/transcribe/batch` - Transcribe multiple files

### Subtitles
- `POST /api/subtitle/transcribe` - Generate subtitles
- `GET /api/subtitle/{id}/srt` - Download SRT file

### Models
- `GET /api/models` - List available models
- `POST /api/models/switch` - Switch to different model

### Vocabulary
- `GET /api/vocabulary` - List vocabulary
- `POST /api/vocabulary` - Add word
- `DELETE /api/vocabulary/{id}` - Delete word

### Other
- `GET /api/history` - Get transcription history
- `GET /api/stats` - Get statistics
- `POST /api/translate` - Convert text

## Tech Stack

- **Backend**: FastAPI, Python
- **AI Models**: HuggingFace Transformers, Whisper
- **Audio Processing**: librosa, soundfile, ffmpeg
- **Frontend**: Vanilla JavaScript, CSS

## Project Structure

```
GujTranscribe/
├── gujarati_asr/
│   ├── main.py          # FastAPI backend
│   ├── web/
│   │   └── index.html   # Frontend UI
│   └── data/            # Data storage
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## Requirements

```
fastapi>=0.100.0
uvicorn>=0.23.0
torch>=2.0.0
transformers>=4.30.0
librosa>=0.10.0
soundfile>=0.12.0
python-multipart>=0.0.6
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## Support

For issues and questions, please open an issue on GitHub.

---

Made with ❤️ for Gujarati language
