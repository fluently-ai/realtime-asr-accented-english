"""Offline ASR adapters. Each call returns a final transcript and raw response."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .local_models import (
        invoke_batch as invoke_local_batch,
    )
    from .local_models import (
        invoke_single as invoke_local_single,
    )
    from .local_models import (
        supports_batch,
        supports_single,
    )
except ImportError:  # direct script execution
    from local_models import (
        invoke_batch as invoke_local_batch,
    )
    from local_models import (
        invoke_single as invoke_local_single,
    )
    from local_models import (
        supports_batch,
        supports_single,
    )


_GOOGLE_AUTH_LOCK = threading.Lock()
_GOOGLE_CREDENTIALS: Any = None
_GOOGLE_ADC_PROJECT: str | None = None
_RATE_LIMIT_LOCK = threading.Lock()
_NEXT_REQUEST_AT: dict[str, float] = {}


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    adapter: str
    model: str
    api_key_env: str | None
    cache_generation: str
    connection: dict[str, Any]
    request: dict[str, Any]

    def cache_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "adapter": self.adapter,
            "model": self.model,
            "cache_generation": self.cache_generation,
            "connection": self.connection,
            "request": self.request,
            "input_profile": "original-file-v1",
        }


def invoke_offline(
    config: ModelConfig,
    *,
    audio_path: Path,
    audio_bytes: bytes,
    credential: str | None,
    timeout_s: float,
) -> dict[str, Any]:
    adapters: dict[str, Callable[..., dict[str, Any]]] = {
        "elevenlabs_file": _elevenlabs,
        "deepgram_file": _deepgram,
        "assemblyai_file": _assemblyai,
        "cartesia_file": _cartesia,
        "openai_file": _openai,
        "google_chirp3": _google,
        "microsoft_fast_file": _microsoft_fast,
        "smallest_file": _smallest,
        "xai_file": _xai,
        "meta_file": _meta,
        "reson8_file": _reson8,
        "gemini_file": _gemini,
        "soniox_file": _soniox,
    }
    if config.api_key_env and not credential:
        raise ProviderError(f"missing credential in {config.api_key_env}")
    local = supports_single(config.adapter)
    if not local:
        try:
            adapter = adapters[config.adapter]
        except KeyError as error:
            raise ValueError(f"unknown adapter: {config.adapter}") from error
    try:
        _wait_for_rate_limit(config)
        raw = (
            invoke_local_single(config, audio_path)
            if local
            else adapter(config, audio_path, audio_bytes, credential, timeout_s)
        )
    except ProviderError:
        raise
    except Exception as error:
        name = type(error).__name__
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        detail = f"{name} HTTP {status}" if isinstance(status, int) else name
        # Dropped connections and timeouts (httpx transport errors) are transient too.
        transport = type(error).__module__.startswith("httpx") and status is None
        retryable = (
            transport
            or status in {408, 409, 425, 429}
            or (isinstance(status, int) and status >= 500)
        )
        raise ProviderError(
            f"{config.provider} offline request failed ({detail})",
            retryable=retryable,
        ) from error
    return {"text": _extract_text(config.adapter, raw), "raw": raw}


def _client(timeout_s: float):
    try:
        import httpx
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("run inference/run.py through uv") from error
    return httpx.Client(timeout=timeout_s, follow_redirects=True)


def _json(response: Any, provider: str) -> dict[str, Any]:
    if not 200 <= response.status_code < 300:
        raise ProviderError(
            f"{provider} HTTP {response.status_code}",
            retryable=response.status_code in {408, 409, 425, 429}
            or response.status_code >= 500,
        )
    try:
        value = response.json()
    except ValueError as error:
        raise ProviderError(f"{provider} returned non-JSON") from error
    if not isinstance(value, dict):
        raise ProviderError(f"{provider} returned non-object JSON")
    return value


def _mime(path: Path) -> str:
    return {
        ".flac": "audio/flac",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
    }.get(path.suffix.lower(), "application/octet-stream")


def _form(value: Any) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def _elevenlabs(config, path, audio, key, timeout):
    request = dict(config.request)
    logging = request.pop("enable_logging", None)
    with _client(timeout) as client:
        response = client.post(
            config.connection["endpoint"],
            headers={"xi-api-key": key},
            params={"enable_logging": _form(logging)} if logging is not None else {},
            data={
                "model_id": config.model,
                **{name: _form(value) for name, value in request.items()},
            },
            files={"file": (path.name, audio, _mime(path))},
        )
    return _json(response, config.provider)


def _deepgram(config, path, audio, key, timeout):
    with _client(timeout) as client:
        response = client.post(
            config.connection["endpoint"],
            headers={"Authorization": f"Token {key}", "Content-Type": _mime(path)},
            params={"model": config.model, **config.request},
            content=audio,
        )
    return _json(response, config.provider)


def _assemblyai(config, _path, audio, key, timeout):
    base = config.connection["api_base"].rstrip("/")
    headers = {"Authorization": key}
    started = time.monotonic()
    with _client(timeout) as client:
        upload = _json(
            client.post(
                f"{base}/v2/upload",
                headers={**headers, "Content-Type": "application/octet-stream"},
                content=audio,
            ),
            config.provider,
        )
        upload_url = upload.get("upload_url")
        if not isinstance(upload_url, str):
            raise ProviderError("AssemblyAI upload returned no URL")
        result = _json(
            client.post(
                f"{base}/v2/transcript",
                headers=headers,
                json={
                    "audio_url": upload_url,
                    "speech_models": [config.model],
                    **config.request,
                },
            ),
            config.provider,
        )
        transcript_id = result.get("id")
        if not isinstance(transcript_id, str):
            raise ProviderError("AssemblyAI returned no transcript id")
        while result.get("status") not in {"completed", "error"}:
            if time.monotonic() - started >= timeout:
                raise ProviderError("AssemblyAI timed out", retryable=True)
            time.sleep(float(config.connection.get("poll_interval_s", 1)))
            result = _json(
                client.get(f"{base}/v2/transcript/{transcript_id}", headers=headers),
                config.provider,
            )
    if result.get("status") == "error":
        raise ProviderError(f"AssemblyAI failed: {result.get('error')}")
    return _redact(result, upload_url)


def _cartesia(config, path, audio, key, timeout):
    form: list[tuple[str, tuple[None, str]]] = [("model", (None, config.model))]
    for name, value in config.request.items():
        values = value if isinstance(value, list) else [value]
        form.extend((name, (None, _form(item))) for item in values)
    with _client(timeout) as client:
        response = client.post(
            config.connection["endpoint"],
            headers={
                "Authorization": f"Bearer {key}",
                "Cartesia-Version": config.connection["api_version"],
            },
            files=[*form, ("file", (path.name, audio, _mime(path)))],
        )
    return _json(response, config.provider)


def _openai(config, path, audio, key, timeout):
    wav = _wav_bytes(path, audio)
    request = {
        name: value for name, value in config.request.items() if name != "input_format"
    }
    with _client(timeout) as client:
        response = client.post(
            config.connection["endpoint"],
            headers={"Authorization": f"Bearer {key}"},
            data={"model": config.model, **request},
            files={"file": (f"{path.stem}.wav", wav, "audio/wav")},
        )
    return _json(response, config.provider)


def _google(config, _path, audio, _key, timeout):
    token, adc_project = _google_access_token()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or adc_project
    if not project:
        raise ProviderError("Google Cloud project is not configured")
    location = config.connection["location"]
    endpoint = (
        f"https://{location}-speech.googleapis.com/v2/projects/{project}/locations/"
        f"{location}/recognizers/_:recognize"
    )
    payload = {
        "config": {
            "autoDecodingConfig": {},
            "languageCodes": [config.request["language_code"]],
            "model": config.model,
            "features": {
                "enableAutomaticPunctuation": config.request[
                    "enable_automatic_punctuation"
                ]
            },
        },
        "content": base64.b64encode(audio).decode("ascii"),
    }
    with _client(timeout) as client:
        response = client.post(
            endpoint,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
    return _json(response, config.provider)


def _smallest(config, _path, audio, key, timeout):
    with _client(timeout) as client:
        response = client.post(
            config.connection["endpoint"],
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/octet-stream",
            },
            params={"model": config.model, **config.request},
            content=audio,
        )
    return _json(response, config.provider)


def _microsoft_fast(config, path, audio, key, timeout):
    endpoint_env = config.connection["endpoint_env"]
    endpoint = os.environ.get(endpoint_env, "").rstrip("/")
    if not endpoint:
        raise ProviderError(f"missing endpoint in {endpoint_env}")
    definition = json.loads(json.dumps(config.request))
    enhanced_mode = definition.setdefault("enhancedMode", {})
    enhanced_mode.update({"enabled": True, "model": config.model})
    with _client(timeout) as client:
        response = client.post(
            f"{endpoint}/speechtotext/transcriptions:transcribe",
            headers={"Ocp-Apim-Subscription-Key": key},
            params={"api-version": config.connection["api_version"]},
            files={
                "audio": (path.name, audio, _mime(path)),
                "definition": (
                    None,
                    json.dumps(definition, separators=(",", ":")),
                    "application/json",
                ),
            },
        )
    return _json(response, config.provider)


def _xai(config, path, audio, key, timeout):
    with _client(timeout) as client:
        response = client.post(
            config.connection["endpoint"],
            headers={"Authorization": f"Bearer {key}"},
            data={
                "model": config.model,
                **{name: _form(value) for name, value in config.request.items()},
            },
            # xAI ignores form fields that follow the file part.
            files={"file": (path.name, audio, _mime(path))},
        )
    return _json(response, config.provider)


def _meta(config, path, audio, key, timeout):
    # Muse accepts only mono 16-bit PCM WAV at 16 or 24 kHz.
    wav = _wav_bytes(path, audio, sample_rate=config.connection["sample_rate_hz"])
    request = {"model": config.model, "audioEncoding": "WAV", **config.request}
    with _client(timeout) as client:
        response = client.post(
            config.connection["endpoint"],
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            files={
                "request": (None, json.dumps(request), "application/json"),
                "audio": (f"{path.stem}.wav", wav, "audio/wav"),
            },
        )
    return _json(response, config.provider)


def _reson8(config, _path, audio, key, timeout):
    # The API has no model parameter; requests run on Reson8's current pipeline.
    with _client(timeout) as client:
        response = client.post(
            config.connection["endpoint"],
            headers={
                "Authorization": f"ApiKey {key}",
                "Content-Type": "application/octet-stream",
            },
            params={name: _form(value) for name, value in config.request.items()},
            content=audio,
        )
    return _json(response, config.provider)


def _gemini(config, path, audio, key, timeout):
    # Inline audio keeps clips out of the Files API store.
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": _mime(path),
                            "data": base64.b64encode(audio).decode("ascii"),
                        }
                    }
                ]
            }
        ],
        "generationConfig": {"audioTranscriptionConfig": config.request},
    }
    with _client(timeout) as client:
        response = client.post(
            f"{config.connection['api_base'].rstrip('/')}/models/{config.model}:generateContent",
            headers={"x-goog-api-key": key},
            json=payload,
        )
    return _json(response, config.provider)


def _soniox(config, path, audio, key, timeout):
    base = config.connection["api_base"].rstrip("/")
    headers = {"Authorization": f"Bearer {key}"}
    started = time.monotonic()
    with _client(timeout) as client:
        upload = _json(
            client.post(
                f"{base}/v1/files",
                headers=headers,
                files={"file": (path.name, audio, _mime(path))},
            ),
            config.provider,
        )
        file_id = upload.get("id")
        if not isinstance(file_id, str):
            raise ProviderError("Soniox upload returned no file id")
        job_id = None
        try:
            job = _json(
                client.post(
                    f"{base}/v1/transcriptions",
                    headers=headers,
                    json={"model": config.model, "file_id": file_id, **config.request},
                ),
                config.provider,
            )
            job_id = job.get("id")
            if not isinstance(job_id, str):
                raise ProviderError("Soniox returned no transcription id")
            while job.get("status") not in {"completed", "error"}:
                if time.monotonic() - started >= timeout:
                    raise ProviderError("Soniox timed out", retryable=True)
                time.sleep(float(config.connection.get("poll_interval_s", 1)))
                job = _json(
                    client.get(f"{base}/v1/transcriptions/{job_id}", headers=headers),
                    config.provider,
                )
            if job.get("status") == "error":
                raise ProviderError(f"Soniox failed: {job.get('error_message')}")
            return _json(
                client.get(
                    f"{base}/v1/transcriptions/{job_id}/transcript", headers=headers
                ),
                config.provider,
            )
        finally:
            # Clips must not stay on the provider; the account also caps stored files.
            if job_id:
                client.delete(f"{base}/v1/transcriptions/{job_id}", headers=headers)
            client.delete(f"{base}/v1/files/{file_id}", headers=headers)


def invoke_batch(config: ModelConfig, audio_paths: list[Path]) -> list[dict[str, Any]]:
    try:
        raw_rows = invoke_local_batch(config, audio_paths)
    except Exception as error:
        raise ProviderError(
            f"{config.provider} offline request failed ({type(error).__name__})"
        ) from error
    return [
        {"text": _extract_text(config.adapter, raw), "raw": raw} for raw in raw_rows
    ]


def _extract_text(adapter: str, raw: dict[str, Any]) -> str:
    if (
        adapter
        in {
            "elevenlabs_file",
            "assemblyai_file",
            "cartesia_file",
            "openai_file",
            "xai_file",
            "reson8_file",
            "soniox_file",
        }
        or supports_single(adapter)
        or supports_batch(adapter)
    ):
        text = raw.get("text")
    elif adapter == "deepgram_file":
        text = _nested(raw, "results", "channels", 0, "alternatives", 0, "transcript")
    elif adapter == "google_chirp3":
        transcripts = [
            _nested(result, "alternatives", 0, "transcript")
            for result in raw.get("results", [])
            if isinstance(result, dict)
        ]
        text = " ".join(item for item in transcripts if isinstance(item, str))
    elif adapter == "smallest_file":
        text = raw.get("transcription")
    elif adapter == "meta_file":
        text = raw.get("transcript")
    elif adapter == "gemini_file":
        # Transcription models answer in audioTranscription parts, not text parts.
        parts = _nested(raw, "candidates", 0, "content", "parts") or []
        pieces = [
            _nested(part, "audioTranscription", "text") or _nested(part, "text")
            for part in parts
        ]
        text = "".join(piece for piece in pieces if isinstance(piece, str))
    elif adapter == "microsoft_fast_file":
        text = " ".join(
            phrase["text"]
            for phrase in raw.get("combinedPhrases", [])
            if isinstance(phrase, dict) and isinstance(phrase.get("text"), str)
        )
    else:  # pragma: no cover - guarded by invoke_offline
        raise ValueError(f"unknown adapter: {adapter}")
    if not isinstance(text, str):
        raise ProviderError(f"{adapter} response has no transcript")
    return text


def _wait_for_rate_limit(config: ModelConfig) -> None:
    requests_per_minute = config.connection.get("requests_per_minute")
    if requests_per_minute is None:
        return
    if not isinstance(requests_per_minute, (int, float)) or requests_per_minute <= 0:
        raise ValueError("requests_per_minute must be positive")
    interval = 60 / float(requests_per_minute)
    with _RATE_LIMIT_LOCK:
        now = time.monotonic()
        scheduled = max(now, _NEXT_REQUEST_AT.get(config.name, now))
        _NEXT_REQUEST_AT[config.name] = scheduled + interval
    if scheduled > now:
        time.sleep(scheduled - now)


def _nested(value: Any, *path: str | int) -> Any:
    for part in path:
        try:
            value = value[part]
        except (KeyError, IndexError, TypeError):
            return None
    return value


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, dict):
        return {key: _redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, str):
        return value.replace(secret, "<upload-url-redacted>")
    return value


def _wav_bytes(path: Path, audio: bytes, sample_rate: int | None = None) -> bytes:
    if path.suffix.lower() == ".wav" and sample_rate is None:
        return audio
    resample = ["-ac", "1", "-ar", str(sample_rate)] if sample_rate else []
    process = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map_metadata",
            "-1",
            *resample,
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if process.returncode or not process.stdout:
        raise ProviderError("ffmpeg could not convert audio to WAV")
    return process.stdout


def _google_access_token() -> tuple[str, str | None]:
    global _GOOGLE_CREDENTIALS, _GOOGLE_ADC_PROJECT
    with _GOOGLE_AUTH_LOCK:
        try:
            import google.auth
            from google.auth.transport.requests import Request
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("run inference/run.py through uv") from error
        try:
            if _GOOGLE_CREDENTIALS is None:
                _GOOGLE_CREDENTIALS, _GOOGLE_ADC_PROJECT = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            if not _GOOGLE_CREDENTIALS.valid:
                _GOOGLE_CREDENTIALS.refresh(Request())
            token = _GOOGLE_CREDENTIALS.token
        except Exception:
            process = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                check=False,
                capture_output=True,
                text=True,
            )
            token = process.stdout.strip() if process.returncode == 0 else None
        if not isinstance(token, str) or not token:
            raise ProviderError("Google ADC returned no access token")
        return token, _GOOGLE_ADC_PROJECT
