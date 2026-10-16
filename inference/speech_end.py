# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.2.6",
#   "silero-vad==6.2.1",
#   "torch==2.13.0",
# ]
# ///
"""Add each clip's Silero VAD end of speech to an input CSV for the realtime runner.

The realtime runner sends the provider's endpoint request once the audio up to
`speech_end_s` has been streamed; latency is timed from that request.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

try:
    from .run import resolve_audio
except ImportError:  # direct `uv run --script inference/speech_end.py`
    from run import resolve_audio

RATE = 16000
SETTINGS = {
    "threshold": 0.5,
    "min_speech_duration_ms": 50,
    "min_silence_duration_ms": 100,
    "speech_pad_ms": 30,  # Silero's default, set explicitly
}


def decode(path: Path) -> np.ndarray:
    process = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(RATE),
            "-f",
            "f32le",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(process.stdout, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = [*reader.fieldnames, "duration_s", "speech_end_s"]
        rows = list(reader)
    torch.set_num_threads(1)
    model = load_silero_vad(onnx=False)
    for row in rows:
        audio = decode(resolve_audio(row["audio_key"], args.audio_root))
        segments = get_speech_timestamps(
            torch.from_numpy(audio.copy()), model, **SETTINGS
        )
        duration = len(audio) / RATE
        # Clips without detected speech end at the last sample.
        end = segments[-1]["end"] / RATE if segments else duration
        row["duration_s"] = f"{duration:.3f}"
        row["speech_end_s"] = f"{min(end, duration):.3f}"
    with args.output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} clips -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
