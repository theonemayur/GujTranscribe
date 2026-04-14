# GujTranscribe - Standalone Windows Application

Gujarati Speech-to-Text application with AI-powered transcription using Whisper models.

## Download

Go to the [Releases](https://github.com/theoneamayur/GujTranscribe/releases) page and download the latest version.

## Quick Start

1. Download and extract the zip file
2. Double-click `Run_GujTranscribe.bat`
3. Wait for the browser to open at http://localhost:8000

That's it! No installation required.

## System Requirements

- Windows 10 or later (64-bit)
- 4GB+ RAM (8GB recommended)
- Internet connection (for first run to download AI models)

## Features

- **6 Whisper Models** - Choose from tiny to large-v3
- **Gujarati ASR** - Optimized for Gujarati language
- **Subtitle Generation** - Create SRT/VTT files
- **Batch Transcription** - Process multiple files
- **Voice Recording** - Record directly in browser
- **Vocabulary Management** - Build custom word lists
- **Word Analysis** - Frequency and statistics
- **Dark/Light Theme** - Modern UI

## Files Included

```
output/
├── GujTranscribe.exe      # Main application
├── gujarati_asr/         # Required files (keep together!)
├── Run_GujTranscribe.bat  # Launcher script
└── README.md
```

## Troubleshooting

### App doesn't start
- Make sure `GujTranscribe.exe` and `gujarati_asr` folder are in the same location
- Right-click `Run_GujTranscribe.bat` → "Run as administrator"

### Browser doesn't open
- Manually open http://localhost:8000

### Model download fails
- Check your internet connection
- Try running as administrator

## License

MIT License

## For Developers

If you want to run from source code:

```bash
# Clone the repository
git clone https://github.com/theoneamayur/GujTranscribe.git

# Install dependencies
cd GujTranscribe/gujarati_asr
pip install -r requirements.txt

# Run the application
python main.py
```
