# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx==0.28.1",
#   "google-auth[requests]==2.56.3",
#   "google-cloud-speech==2.41.0",
#   "python-dotenv==1.1.0",
#   "websockets==15.0.1",
# ]
# ///
"""Stream clips to realtime ASR models at 1x speed; record transcripts and latency.

The protocol follows AA-WER Streaming. Audio goes out in 100 ms chunks at real-time
speed. Once the audio up to the clip's Silero VAD end of speech (`speech_end_s`, see
speech_end.py) has been sent, the runner sends the model's endpoint request and keeps
streaming the rest of the clip, then silence. The scored transcript is the text known at
the first final transcript after that point: every final segment up to and including it.
A stream without any final transcript is retried; if every attempt ends that way, the
empty transcript stands.
Output rows follow run.py, so score.py scores them unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .realtime_providers import ADAPTERS, Stream
    from .run import (
        DEFAULT_CACHE,
        DEFAULT_CONFIG,
        Input,
        ModelConfig,
        cache_file,
        load_cache,
        load_configs,
        read_inputs,
        write_json,
        write_jsonl,
    )
except ImportError:  # direct `uv run --script inference/run_realtime.py`
    from realtime_providers import ADAPTERS, Stream
    from run import (
        DEFAULT_CACHE,
        DEFAULT_CONFIG,
        Input,
        ModelConfig,
        cache_file,
        load_cache,
        load_configs,
        read_inputs,
        write_json,
        write_jsonl,
    )

PROTOCOL = {"version": "aa-wer-streaming-v1", "chunk_ms": 100, "final_timeout_s": 15.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--audio-root", type=Path, action="append", required=True)
    parser.add_argument("--env-file", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--workers", type=int, default=32, help="concurrent streams per model"
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for path in args.env_file:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
    configs = load_configs(args.config)
    try:
        selected = [configs[name] for name in args.model]
    except KeyError as error:
        parser.error(f"unknown model: {error.args[0]}")
    for config in selected:
        if config.adapter not in ADAPTERS:
            parser.error(f"{config.name} is not a realtime model")
    inputs = read_inputs(args.input, args.audio_root, limit=args.limit)
    if any("speech_end_s" not in item.fields for item in inputs):
        parser.error(
            "input CSV needs speech_end_s; add it with inference/speech_end.py"
        )
    results = asyncio.run(
        run(inputs, selected, args.cache_dir, args.workers, args.retries, args.execute)
    )
    write_jsonl(args.output, results)
    errors = sum(row["status"] == "error" for row in results)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "rows": len(results),
                "errors": errors,
            }
        )
    )
    return 2 if errors else 0


async def run(
    inputs: list[Input],
    configs: list[ModelConfig],
    cache_dir: Path,
    workers: int,
    retries: int,
    execute: bool,
) -> list[dict[str, Any]]:
    keyed = [
        dataclasses.replace(c, connection={**c.connection, "protocol": PROTOCOL})
        for c in configs
    ]
    jobs, results = [], {}
    for config in keyed:
        for item in inputs:
            path = cache_file(cache_dir, config, item.audio_sha256)
            cached = load_cache(path, config, item.audio_sha256)
            if cached is None:
                jobs.append((item, config, path))
            else:
                results[item.audio_key, config.name] = row(
                    item, config, cached, cache_hit=True
                )
    if jobs and not execute:
        raise SystemExit(f"{len(jobs)} uncached streams; pass --execute")
    for config in keyed:
        if config.api_key_env and not os.environ.get(config.api_key_env):
            raise SystemExit(f"missing {config.api_key_env}")
    limits = {config.name: asyncio.Semaphore(workers) for config in keyed}
    done = 0

    async def job(item: Input, config: ModelConfig, path: Path) -> None:
        nonlocal done
        async with limits[config.name]:
            try:
                cached = await stream_clip(item, config, retries)
                write_json(path, cached)
                results[item.audio_key, config.name] = row(
                    item, config, cached, cache_hit=False
                )
            except Exception as error:  # noqa: BLE001 - one failed stream must not stop the run
                results[item.audio_key, config.name] = {
                    "audio_key": item.audio_key,
                    "provider": config.provider,
                    "model": config.name,
                    "status": "error",
                    "error": f"{config.provider} stream failed ({type(error).__name__}: {error})"[
                        :300
                    ],
                    "text": "",
                    "cache_hit": False,
                    "input": item.fields,
                }
        done += 1
        if done % 50 == 0 or done == len(jobs):
            print(
                f"completed {done}/{len(jobs)} uncached streams",
                file=sys.stderr,
                flush=True,
            )

    await asyncio.gather(*(job(*entry) for entry in jobs))
    return [results[item.audio_key, config.name] for config in keyed for item in inputs]


_NEXT_SESSION_AT: dict[str, float] = {}


async def pace(config: ModelConfig) -> None:
    """Space out session starts for providers that cap new sessions per minute."""
    per_minute = config.connection.get("sessions_per_minute")
    if not per_minute:
        return
    now = time.monotonic()
    start = max(now, _NEXT_SESSION_AT.get(config.name, now))
    _NEXT_SESSION_AT[config.name] = start + 60 / float(per_minute)
    await asyncio.sleep(start - now)


async def stream_clip(item: Input, config: ModelConfig, retries: int) -> dict[str, Any]:
    rate = int(config.connection.get("sample_rate_hz", 16000))
    # Decode off the event loop so it never delays other streams' chunks or timestamps.
    pcm = await asyncio.to_thread(pcm16, item.audio_path, rate)
    key = os.environ.get(config.api_key_env) if config.api_key_env else None
    for attempt in range(1, retries + 1):
        stream = Stream(
            pcm=pcm,
            rate=rate,
            speech_end_s=float(item.fields["speech_end_s"]),
            chunk_ms=PROTOCOL["chunk_ms"],
            final_timeout_s=PROTOCOL["final_timeout_s"],
        )
        await pace(config)
        try:
            await asyncio.wait_for(
                ADAPTERS[config.adapter](config, key, stream),
                timeout=len(pcm) / (2 * rate) + PROTOCOL["final_timeout_s"] + 30,
            )
            # A session that never finalizes anything usually failed like a dropped
            # connection, so retry it; silent clips keep their empty last attempt.
            if attempt < retries and not any(kind == "final" for kind, *_ in stream.events):
                raise RuntimeError("stream ended without a final transcript")
            break
        except Exception:
            if attempt == retries:
                raise
            await asyncio.sleep(min(2**attempt, 15))
    text, timing = stream.result()
    return {
        "audio_sha256": item.audio_sha256,
        "config": config.cache_dict(),
        "text": text,
        "raw": {"timing": timing, "events": stream.events},
    }


def row(
    item: Input, config: ModelConfig, cached: dict[str, Any], *, cache_hit: bool
) -> dict[str, Any]:
    return {
        "audio_key": item.audio_key,
        "provider": config.provider,
        "model": config.name,
        "status": "ok",
        "error": None,
        "text": cached["text"],
        "cache_hit": cache_hit,
        "realtime": cached["raw"]["timing"],
        "input": item.fields,
    }


def pcm16(path: Path, rate: int) -> bytes:
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
            str(rate),
            "-f",
            "s16le",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return process.stdout


if __name__ == "__main__":
    raise SystemExit(main())
