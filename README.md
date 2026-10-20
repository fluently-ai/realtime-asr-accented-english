# Real-Time Speech Recognition on Accented English: inference code

The code that produced every transcript in our benchmark of real-time speech recognition on
accented English: 16 offline profiles (file APIs and two open models) and 22 real-time profiles
(streaming APIs). Every model ID and request option is pinned in
[`inference/models.toml`](inference/models.toml). Results and methodology are in the
[report](https://getfluently.app/research/asr). The benchmark audio and transcripts, text
normalization and WER scoring are not part of this repository.

```text
inference/
├── models.toml            # all 38 profiles: model IDs, endpoints, request options
├── run.py                 # offline runner (file APIs)
├── providers.py           # offline adapters, one per API
├── run_transformers.py    # offline runner for the open models (Parakeet, Qwen3-ASR)
├── local_models.py        # Transformers runtimes for those models
├── speech_end.py          # adds the Silero VAD end of speech that real-time runs need
├── run_realtime.py        # real-time runner: paced streaming, endpointing, timing
├── realtime_providers.py  # real-time adapters, one per streaming API
└── .env.example           # credential variable names
```

## Setup

Install [uv](https://docs.astral.sh/uv/) and `ffmpeg`. Each script declares its pinned
dependencies inline, so `uv run --script` installs them on first use. There is no other setup.

Copy `inference/.env.example` to a private `.env` and fill in the keys for the providers you
want; pass it with `--env-file`. Keys are read only from the environment and are never written to
outputs or the cache. Google Speech-to-Text V2 (`chirp_3`, `chirp_3-realtime`) authenticates with
Application Default Credentials (`gcloud auth application-default login`) and needs
`GOOGLE_CLOUD_PROJECT`. `azure-stt-realtime` uses Microsoft's Speech SDK, which on Linux needs
glibc 2.29+ and OpenSSL 3.

## Input

A CSV with a unique `audio_key` column: the path of each clip relative to an `--audio-root`
directory. Any other columns are copied into each output row under `input` and are never sent to
a provider. Real-time runs also need `speech_end_s` (see below).

## Offline runs

```bash
uv run --script inference/run.py \
  --input clips.csv --audio-root /path/to/audio --env-file .env \
  --model nova-3 --model scribe_v2 \
  --workers 32 --output offline.jsonl --execute
```

The 16 offline profiles:

| Profile | Provider | Profile | Provider |
|---|---|---|---|
| `scribe_v2` | ElevenLabs | `chirp_3` | Google Cloud |
| `mai-transcribe-2` | Microsoft | `parakeet-tdt-0.6b-v3` | NVIDIA (open model) |
| `resonant-1` | Reson8 | `muse-voice-transcribe-1.0` | Meta |
| `qwen3-asr-1.7b` | Qwen (open model) | `nova-3` | Deepgram |
| `grok-voice-transcribe-2.0` | xAI | `stt-async-v5` | Soniox |
| `gemini-3.5-transcribe` | Google Gemini | `gpt-transcribe` | OpenAI |
| `universal-3-5-pro` | AssemblyAI | `voxtral-mini-2602` | Mistral |
| `ink-whisper` | Cartesia | `pulse-pro` | Smallest AI |

- Every API gets each clip once, whole. The request options (language hint, verbatim or
  formatting switches, diarization off) are in `models.toml`; `providers.py` shows how each
  request is built (raw file upload, multipart, or WAV conversion with `ffmpeg` where an API
  requires a format or sample rate). Two profiles get resampled audio because their providers
  ask for it: `pulse-pro` 16 kHz mono and `muse-voice-transcribe-1.0` 24 kHz mono.
- The two open models run locally with greedy decoding:
  `uv run --script inference/run_transformers.py ... --model parakeet-tdt-0.6b-v3`.
- Requests that fail with a timeout, a transport error, or HTTP 408, 409, 425, 429 or 5xx are
  retried with backoff; each request is attempted up to `--retries` times (default 3). Profiles
  with a provider rate limit carry `requests_per_minute` in `models.toml`, and the runner paces
  requests to it.
- Concurrency (`--workers`) does not change offline outputs. We used 16–48 workers, fewer where a
  provider limits concurrent jobs.

## Real-time runs

Real-time runs follow a protocol adapted from Artificial Analysis's
[streaming methodology](https://artificialanalysis.ai/methodology/speech-to-text). The
differences:

- 100 ms audio chunks instead of 20 ms.
- After the endpoint, the runner waits up to 15 s for a final transcript; Artificial Analysis
  waits 2 s and then falls back to partials.
- Several providers keep their default endpointing, while Artificial Analysis configures
  endpointing so that only the forced endpoint finalizes.

**1. Find the end of speech.** `speech_end.py` runs Silero VAD (`silero-vad==6.2.1`, 16 kHz,
threshold 0.5, minimum speech 50 ms, minimum silence 100 ms, `speech_pad_ms` 30) and adds two
columns: `duration_s` and `speech_end_s`, the end of the last speech segment (the clip end if no
speech is found). Like the runners, it accepts several `--audio-root` directories.

```bash
uv run --script inference/speech_end.py \
  --input clips.csv --audio-root /path/to/audio --output clips-vad.csv
```

**2. Stream.** `run_realtime.py` streams each clip in its own session:

```bash
uv run --script inference/run_realtime.py \
  --input clips-vad.csv --audio-root /path/to/audio --env-file .env \
  --model nova-3-realtime --workers 48 --output realtime.jsonl --execute
```

- **Audio.** `ffmpeg` decodes each clip to mono 16-bit PCM at the profile's rate (16 kHz;
  24 kHz for `gpt-live-transcribe`, `gpt-realtime-whisper` and
  `muse-voice-transcribe-1.0-realtime`). The runner sends
  100 ms chunks at 1x speed: a chunk goes out once its last sample would have been spoken.
- **Endpoint.** Right after the chunk that completes `speech_end_s`, the runner sends the
  provider's endpoint (finalize) request and streams the rest of the clip. After the clip it
  streams silence until the first final transcript after the endpoint arrives or 15 s have
  passed since the endpoint. Then it closes the session and keeps reading for up to 5 s, except
  for `muse-voice-transcribe-1.0-realtime`, whose session closes itself after `endStream`.
- **Scored text.** Every final segment up to and including the first final after the endpoint.
  Later finals are recorded but not scored.
- **Timing.** `time_to_final_s` and `time_to_first_partial_s` start when the endpoint request
  is sent, which is when Artificial Analysis starts its timer. `endpoint_at_s` and
  `speech_end_s` are stored too.
- **Retries.** A session that fails (connection or provider error, or a timeout at the audio
  length plus 45 s) or ends without any final transcript is retried with backoff; each clip is
  attempted up to `--retries` times (default 3). If every attempt ends without a final, the
  empty transcript stands (for example on silent clips).
- **Events.** The cache keeps every partial and final with its arrival time, so other cut-offs
  can be computed later without streaming again.

Endpoint and close messages per profile (see `realtime_providers.py`):

| Profile | Endpoint request | End of session |
|---|---|---|
| `nova-3-realtime` | `Finalize` | `CloseStream` |
| `scribe_v2_realtime` | empty chunk with `commit: true` (manual commit) | close socket |
| `gpt-live-transcribe` | `input_audio_buffer.commit` (no server VAD) | close socket |
| `grok-voice-transcribe-2.0-realtime` | `finalize` | `audio.done` |
| `universal-3-5-pro-realtime` | `ForceEndpoint` | `Terminate` |
| `stt-rt-v5` | `finalize` | empty frame |
| `ink-2-realtime` | `finalize` | `close` |
| `pulse-realtime` | `finalize` | `close_stream` |
| `resonant-1-realtime` | `flush_request` | close socket |
| `voxtral-mini-transcribe-realtime-2602` | `input_audio.flush` | `input_audio.end` |
| `muse-voice-transcribe-1.0-realtime` | `endStream` (push-to-talk ends the turn and input) | server closes |
| `gemini-3.5-transcribe-live` | `audioStreamEnd` | close socket |
| `chirp_3-realtime` | none; the API finalizes on its own endpointing | half-close the stream |
| `mai-transcribe-2-realtime` | `input_audio_buffer.commit` (no server VAD) | close socket |
| `nemotron-3-asr-streaming-realtime` | `input_audio_buffer.commit` (no server VAD) | close socket |
| `universal-3-6-pro-realtime` | `ForceEndpoint` | `Terminate` |
| `flux-general-en-realtime` | `ForceEndTurn` (`eot_threshold = 1.0`: Flux's own end-of-turn is off) | `CloseStream` |
| `linden-1-realtime` | `ForceEndOfUtterance` at the audio sent so far (external turn detection) | `EndOfStream` |
| `azure-stt-realtime` | `PushAudioInputStream.commit()` (Speech SDK, `setfeature=forcecommit`) | close the push stream |
| `gpt-realtime-whisper` | `input_audio_buffer.commit` (no server VAD) | close socket |
| `nari-qwen3-asr-realtime`, `nari-qwen3-asr-fast-realtime` | `input_audio_buffer.commit` (no turn detection) | close socket |

Concurrent sessions per profile in our run (all 15 profiles streamed at once from one 16-vCPU
cloud machine in the US):

| Sessions | Profiles |
|---|---|
| 48 | `nova-3-realtime`, `grok-voice-transcribe-2.0-realtime`, `pulse-realtime`, `muse-voice-transcribe-1.0-realtime` |
| 32 | `gpt-live-transcribe`, `mai-transcribe-2-realtime` |
| 24 | `scribe_v2_realtime`, `universal-3-5-pro-realtime` (also capped at 90 new sessions per minute) |
| 16–18 | `ink-2-realtime`, `gemini-3.5-transcribe-live`, `nemotron-3-asr-streaming-realtime`, `resonant-1-realtime` (18) |
| 8–10 | `stt-rt-v5` (10), `voxtral-mini-transcribe-realtime-2602`, `chirp_3-realtime` |

Five profiles were added later and streamed together from a second 16-vCPU cloud machine in the
US: `flux-general-en-realtime` (48 sessions), `linden-1-realtime` (40), `azure-stt-realtime` (32),
`gpt-realtime-whisper` (32) and `universal-3-6-pro-realtime` (24, 90 new sessions per minute). On that
machine, three earlier profiles rerun on 300 clips matched the first run within 0.1 WER points and
25 ms of median time to final.

The two Nari Labs profiles (`nari-qwen3-asr-realtime`, `nari-qwen3-asr-fast-realtime`) were
streamed last from a third, 2-vCPU cloud machine in the US with 10 sessions each (Nari allows 20
connections per account); the first 838 clips of `nari-qwen3-asr-fast-realtime` ran with 2
sessions. The same three-profile control on that machine matched the first run within 0.12 WER
points; median time to final was 3 and 6 ms higher, and 47 ms lower for
`universal-3-5-pro-realtime`.

## Output

Both runners write one JSONL row per clip and profile: `audio_key`, `provider`, `model` (the
profile name), `status`, `error`, `text`, `cache_hit`, and the untouched input columns under
`input`. Successful real-time rows add `realtime` with `speech_end_s`, `endpoint_at_s`,
`final_after_endpoint`, `time_to_final_s` and `time_to_first_partial_s`; error rows have no
`realtime` field.

Results are cached under `.cache/inference/<profile>/` (set another directory with
`--cache-dir`), keyed by the audio's SHA-256 and the full profile (every request option, plus
the streaming protocol for real-time runs), so reruns only call providers for missing clips. A
cached real-time result whose `speech_end_s` differs from the input's is streamed again.
Without `--execute` a runner stops before making any paid call and reports how many are missing.

## License

The code is released under the [MIT License](LICENSE). Each provider's API and each open model
keeps its own terms of use.
