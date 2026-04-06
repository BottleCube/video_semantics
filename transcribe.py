#!/usr/bin/env python3
"""Transcribe a video/audio file with Whisper and save the result as JSON."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel

WHISPER_MODEL = "large-v3"


def has_audio_stream(video_path: str) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def extract_audio(video_path: str, output_path: str) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", output_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}")


def transcribe(audio_path: str, model_size: str, language: str | None) -> list[tuple[float, str]]:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    kwargs = {"language": language} if language else {}
    segments, _ = model.transcribe(audio_path, **kwargs)
    return [(seg.start, seg.text.strip()) for seg in segments]


def main():
    parser = argparse.ArgumentParser(description="Transcribe a video/audio file and save result as JSON")
    parser.add_argument("video", help="Path to video or audio file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument("--whisper-model", default=WHISPER_MODEL, help=f"Whisper model size (default: {WHISPER_MODEL})")
    parser.add_argument("--language", help="Audio language code e.g. ja, en (default: auto-detect)")
    args = parser.parse_args()

    if not has_audio_stream(args.video):
        print("No audio track found — nothing to transcribe.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Extracting audio...")
        audio_path = os.path.join(tmpdir, "audio.wav")
        extract_audio(args.video, audio_path)

        print(f"Transcribing ({args.whisper_model})...")
        segments = transcribe(audio_path, args.whisper_model, args.language)

    output = json.dumps(segments, ensure_ascii=False, indent=2)
    Path(args.output).write_text(output, encoding="utf-8")
    print(f"Saved {len(segments)} segments to {args.output}")


if __name__ == "__main__":
    main()
