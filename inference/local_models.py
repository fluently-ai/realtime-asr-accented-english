"""Lazy local ASR runtimes, registered by adapter name."""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

_MODELS: dict[str, tuple[Any, Any, str]] = {}
_LOCKS: dict[str, threading.Lock] = {}
_INIT_LOCK = threading.Lock()


def supports_single(adapter: str) -> bool:
    return adapter in _SINGLE_ADAPTERS


def supports_batch(adapter: str) -> bool:
    return adapter in _BATCH_ADAPTERS


def invoke_single(config: Any, audio_path: Path) -> dict[str, Any]:
    try:
        adapter = _SINGLE_ADAPTERS[config.adapter]
    except KeyError as error:
        raise ValueError(f"unknown local adapter: {config.adapter}") from error
    return adapter(config, audio_path)


def invoke_batch(config: Any, audio_paths: list[Path]) -> list[dict[str, Any]]:
    try:
        adapter = _BATCH_ADAPTERS[config.adapter]
    except KeyError as error:
        raise ValueError(f"unknown local batch adapter: {config.adapter}") from error
    return adapter(config, audio_paths)


def _nvidia_transformers(config: Any, path: Path) -> dict[str, Any]:
    with _model_lock(config.name):
        import torch
        from transformers import AutoModelForTDT, AutoProcessor

        loaded = _MODELS.get(config.name)
        if loaded is None:
            device = _device(torch, config.request["device"])
            dtype = getattr(torch, config.request["dtype"])
            processor = AutoProcessor.from_pretrained(config.model)
            model = AutoModelForTDT.from_pretrained(config.model, dtype=dtype)
            model = model.to(device).eval()
            loaded = (processor, model, device)
            _MODELS[config.name] = loaded
        processor, model, device = loaded
        samples = _read_audio(path, int(config.request["sampling_rate"]))
        inputs = processor(
            samples,
            sampling_rate=int(config.request["sampling_rate"]),
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            output = model.generate(**inputs, return_dict_in_generate=True)
        text = processor.decode(output.sequences, skip_special_tokens=True)
        if isinstance(text, list):
            text = text[0]
    return {"text": text, "runtime": {"engine": "transformers", "device": device}}


def _qwen_transformers(config: Any, path: Path) -> dict[str, Any]:
    with _model_lock(config.name):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        loaded = _MODELS.get(config.name)
        if loaded is None:
            device = _device(torch, config.request["device"])
            dtype = getattr(torch, config.request["dtype"])
            processor = AutoProcessor.from_pretrained(config.model)
            model = AutoModelForMultimodalLM.from_pretrained(config.model, dtype=dtype)
            model = model.to(device).eval()
            loaded = (processor, model, device)
            _MODELS[config.name] = loaded
        processor, model, device = loaded
        samples = _read_audio(path, int(config.request["sampling_rate"]))
        inputs = processor.apply_transcription_request(
            audio=samples,
            language=config.request["language"],
        ).to(model.device, model.dtype)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(config.request["max_new_tokens"]),
                do_sample=bool(config.request["do_sample"]),
            )
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        text = processor.decode(generated, return_format="transcription_only")[0]
    return {"text": text, "runtime": {"engine": "transformers", "device": device}}


_SINGLE_ADAPTERS = {
    "nvidia_transformers": _nvidia_transformers,
    "qwen_transformers": _qwen_transformers,
}

_BATCH_ADAPTERS: dict[str, Any] = {}


def _model_lock(model: str) -> threading.Lock:
    with _INIT_LOCK:
        return _LOCKS.setdefault(model, threading.Lock())


def _device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _read_audio(path: Path, sampling_rate: int):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    samples, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if source_rate != sampling_rate:
        divisor = math.gcd(source_rate, sampling_rate)
        samples = resample_poly(
            samples, sampling_rate // divisor, source_rate // divisor
        )
    return np.asarray(samples, dtype=np.float32)
