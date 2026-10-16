# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx==0.28.1",
#   "google-auth[requests]==2.56.3",
#   "python-dotenv==1.1.0",
# ]
# ///
"""Run offline ASR models for a CSV and write final transcripts as JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
import tomllib
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .providers import (
        ModelConfig,
        ProviderError,
        invoke_batch,
        invoke_offline,
        supports_batch,
    )
except ImportError:  # direct `uv run --script inference/run.py`
    from providers import (
        ModelConfig,
        ProviderError,
        invoke_batch,
        invoke_offline,
        supports_batch,
    )

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).with_name("models.toml")
DEFAULT_CACHE = BENCHMARK_ROOT / ".cache" / "inference"


@dataclass(frozen=True)
class Input:
    audio_key: str
    fields: dict[str, str]
    audio_path: Path
    audio_sha256: str


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
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--batch-chunk-size",
        type=int,
        help="Split local batch adapters into restart-safe chunks",
    )
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if len(set(args.model)) != len(args.model):
        parser.error("--model values must be unique")
    if (
        args.workers < 1
        or args.retries < 1
        or args.timeout_s <= 0
        or (args.batch_chunk_size is not None and args.batch_chunk_size < 1)
    ):
        parser.error("workers, retries, and timeout must be positive")

    for path in args.env_file:
        if not path.is_file():
            parser.error(f"env file not found: {path}")
        from dotenv import load_dotenv

        load_dotenv(path, override=False)

    configs = load_configs(args.config)
    try:
        selected = [configs[name] for name in args.model]
    except KeyError as error:
        parser.error(f"unknown model: {error.args[0]}")
    for config in selected:
        if config.adapter.endswith("_realtime"):
            parser.error(f"{config.name} is a realtime model; use run_realtime.py")
    rows = read_inputs(args.input, args.audio_root, limit=args.limit)
    results = run(
        rows,
        selected,
        args.cache_dir,
        args.timeout_s,
        args.retries,
        args.workers,
        args.execute,
        batch_chunk_size=args.batch_chunk_size,
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


def load_configs(path: Path) -> dict[str, ModelConfig]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    models = raw.get("models")
    if not isinstance(models, dict):
        raise TypeError("models.toml must contain [models]")
    result = {}
    for name, value in models.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise TypeError("invalid model config")
        model = _string(value, "model")
        result[name] = ModelConfig(
            name=name,
            provider=_string(value, "provider"),
            adapter=_string(value, "adapter"),
            model=model,
            api_key_env=_optional_string(value, "api_key_env"),
            cache_generation=_string(value, "cache_generation"),
            connection=_dict(value, "connection"),
            request=_dict(value, "request"),
        )
    return result


def read_inputs(
    input_path: Path, audio_roots: list[Path], *, limit: int | None
) -> list[Input]:
    with input_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "audio_key" not in reader.fieldnames:
            raise ValueError("input CSV must contain audio_key")
        rows = [dict(row) for row in reader]
    if limit is not None:
        rows = rows[:limit]
    seen = set()
    result = []
    for fields in rows:
        audio_key = fields.get("audio_key", "").strip()
        relative = Path(audio_key)
        if (
            not audio_key
            or audio_key in seen
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ValueError(f"invalid or duplicate audio_key: {audio_key!r}")
        seen.add(audio_key)
        audio_path = resolve_audio(audio_key, audio_roots)
        result.append(
            Input(
                audio_key=audio_key,
                fields=fields,
                audio_path=audio_path,
                audio_sha256=file_sha256(audio_path),
            )
        )
    return result


def resolve_audio(audio_key: str, audio_roots: list[Path]) -> Path:
    relative = Path(audio_key)
    for root in audio_roots:
        resolved = root.resolve()
        for candidate in (resolved / relative, resolved / relative.name):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"audio not found for {audio_key}")


def run(
    inputs: list[Input],
    configs: list[ModelConfig],
    cache_dir: Path,
    timeout_s: float,
    retries: int,
    workers: int,
    execute: bool,
    *,
    batch_chunk_size: int | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * (len(inputs) * len(configs))
    planned: list[tuple[int, Input, ModelConfig, Path]] = []
    for input_index, item in enumerate(inputs):
        for config_index, config in enumerate(configs):
            index = input_index * len(configs) + config_index
            path = cache_file(cache_dir, config, item.audio_sha256)
            cached = load_cache(path, config, item.audio_sha256)
            if cached is None:
                planned.append((index, item, config, path))
            else:
                results[index] = result_row(item, config, cached, cache_hit=True)
    if planned and not execute:
        raise SystemExit(f"{len(planned)} uncached requests; pass --execute")
    for _index, _item, config, _path in planned:
        if config.api_key_env and not os.environ.get(config.api_key_env):
            raise SystemExit(f"missing {config.api_key_env}")
        endpoint_env = config.connection.get("endpoint_env")
        if isinstance(endpoint_env, str) and not os.environ.get(endpoint_env):
            raise SystemExit(f"missing {endpoint_env}")

    total_planned = len(planned)
    completed = 0
    batch_planned = [entry for entry in planned if supports_batch(entry[2].adapter)]
    planned = [entry for entry in planned if not supports_batch(entry[2].adapter)]
    batch_configs = {entry[2].name: entry[2] for entry in batch_planned}
    for config in batch_configs.values():
        entries = [entry for entry in batch_planned if entry[2] == config]
        chunk_size = batch_chunk_size or len(entries)
        for start in range(0, len(entries), chunk_size):
            chunk = entries[start : start + chunk_size]
            try:
                batch = invoke_batch(config, [entry[1].audio_path for entry in chunk])
                if len(batch) != len(chunk):
                    raise ProviderError("local model returned an incomplete batch")
                for (index, item, _, path), result in zip(chunk, batch):
                    cached = {
                        "audio_sha256": item.audio_sha256,
                        "config": config.cache_dict(),
                        "text": result["text"],
                        "raw": result["raw"],
                    }
                    write_json(path, cached)
                    results[index] = result_row(item, config, cached, cache_hit=False)
            except ProviderError as error:
                for index, item, _, _path in chunk:
                    results[index] = {
                        "audio_key": item.audio_key,
                        "provider": config.provider,
                        "model": config.name,
                        "status": "error",
                        "error": str(error),
                        "text": "",
                        "cache_hit": False,
                        "input": item.fields,
                    }
            completed += len(chunk)
            print(
                f"completed {completed}/{total_planned} uncached requests",
                file=sys.stderr,
                flush=True,
            )

    def request(item: Input, config: ModelConfig, path: Path) -> dict[str, Any]:
        cached = call_model(item, config, timeout_s, retries)
        write_json(path, cached)
        return cached

    future_map: dict[Future[dict[str, Any]], tuple[int, Input, ModelConfig]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, item, config, path in planned:
            future_map[executor.submit(request, item, config, path)] = (
                index,
                item,
                config,
            )
        for future in as_completed(future_map):
            index, item, config = future_map[future]
            try:
                cached = future.result()
                results[index] = result_row(item, config, cached, cache_hit=False)
            except ProviderError as error:
                results[index] = {
                    "audio_key": item.audio_key,
                    "provider": config.provider,
                    "model": config.name,
                    "status": "error",
                    "error": str(error),
                    "text": "",
                    "cache_hit": False,
                    "input": item.fields,
                }
            completed += 1
            print(
                f"completed {completed}/{total_planned} uncached requests",
                file=sys.stderr,
                flush=True,
            )
    return [row for row in results if row is not None]


def call_model(
    item: Input, config: ModelConfig, timeout_s: float, retries: int
) -> dict[str, Any]:
    credential = os.environ.get(config.api_key_env) if config.api_key_env else None
    for attempt in range(1, retries + 1):
        try:
            result = invoke_offline(
                config,
                audio_path=item.audio_path,
                audio_bytes=item.audio_path.read_bytes(),
                credential=credential,
                timeout_s=timeout_s,
            )
            return {
                "audio_sha256": item.audio_sha256,
                "config": config.cache_dict(),
                "text": result["text"],
                "raw": result["raw"],
            }
        except ProviderError as error:
            if not error.retryable or attempt == retries:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))
    raise AssertionError("unreachable")


def result_row(
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
        "input": item.fields,
    }


def cache_file(cache_dir: Path, config: ModelConfig, audio_sha256: str) -> Path:
    key = hashlib.sha256(
        json.dumps(
            {"audio_sha256": audio_sha256, "config": config.cache_dict()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return cache_dir / config.name / f"{key}.json"


def load_cache(
    path: Path, config: ModelConfig, audio_sha256: str
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or (
        value.get("audio_sha256") != audio_sha256
        or value.get("config") != config.cache_dict()
        or not isinstance(value.get("text"), str)
        or not isinstance(value.get("raw"), dict)
    ):
        raise ValueError(f"invalid cache file: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_bytes(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode(),
    )


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise TypeError(f"{key} must be a non-empty string")
    return result


def _optional_string(value: dict[str, Any], key: str) -> str | None:
    result = value.get(key)
    if result is not None and (not isinstance(result, str) or not result):
        raise TypeError(f"{key} must be a non-empty string when set")
    return result


def _dict(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key, {})
    if not isinstance(result, dict):
        raise TypeError(f"{key} must be an object")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
