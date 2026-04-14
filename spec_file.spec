# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

block_cipher = None

project_root = Path(SPECPATH).parent
data_dir = project_root / "gujarati_asr" / "data"
web_dir = project_root / "gujarati_asr" / "web"

a = Analysis(
    ['main.py'],
    pathex=[project_root / "gujarati_asr"],
    binaries=[],
    datas=[
        (str(web_dir), 'web'),
        (str(data_dir), 'data'),
    ],
    hiddenimports=[
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi',
        'starlette',
        'pydantic',
        'transformers',
        'torch',
        'librosa',
        'soundfile',
        'numpy',
        'scipy',
        'sklearn',
        'tokenizers',
        'huggingface_hub',
        'accelerate',
        'safetensors',
        'regex',
        'requests',
        'filelock',
        'fsspec',
        'packaging',
        'pyyaml',
        'typer',
        'typing_extensions',
        'jinja2',
        'python_multipart',
        'anyio',
        'sniffio',
        'idna',
        'certifi',
        'charset_normalizer',
        'urllib3',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GujTranscribe',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GujTranscribe',
)
