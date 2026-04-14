# Gujarati + Gujlish ASR Model

This repository contains scripts to fine-tune OpenAI Whisper for Gujarati (and Gujlish) speech-to-text.

## Files

- `download_data.py`: Downloads a subset of Common Voice Gujarati, resamples audio to 16kHz, and creates a manifest JSONL.
- `requirements.txt`: Python dependencies.
- `train_whisper.py`: Fine-tuning script using Hugging Face Transformers.

## Steps

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download and prepare data**
   ```bash
   python download_data.py
   ```
   This creates `cv_gujarati/` with wav files and `manifest.jsonl`.

3. **Fine-tune Whisper**
   ```bash
   python train_whisper.py \
       --manifest cv_gujarati/manifest.jsonl \
       --output_dir ./whisper_guj_finetuned \
       --model_name_or_path openai/whisper-small \
       --language gu \
       --max_steps 20000 \
       --per_device_train_batch_size 8 \
       --gradient_accumulation_steps 2 \
       --learning_rate 1e-5 \
       --warmup_steps 500 \
       --fp16
   ```
   Adjust parameters as needed for your hardware and dataset size.

4. **Use the model**
   After training, load the model from `./whisper_guj_finetuned` with the WhisperProcessor for transcription.

## Notes

- The download script currently takes 2000 samples from Common Voice Gujarati. Increase `max_samples` for more data.
- For Gujlish (code-mixed Gujarati-English), you may need to add your own data or augment the dataset with Gujlish transcripts.
- Evaluation: The training script reports Word Error Rate (WER) on a validation split.

Happy training!