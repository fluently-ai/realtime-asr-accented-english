"""Realtime ASR adapters. Each streams one clip per session: over a WebSocket, or gRPC for
Chirp and the Speech SDK for Azure.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import websockets


@dataclass
class Stream:
    """One clip: paced audio out, transcript events in, timed against the endpoint."""

    pcm: bytes
    rate: int
    speech_end_s: float
    chunk_ms: int = 100
    final_timeout_s: float = 15.0
    events: list[list[Any]] = field(default_factory=list)
    _start: float = 0.0
    _endpoint_at: float | None = None
    _final: asyncio.Event = field(default_factory=asyncio.Event)

    def now(self) -> float:
        return round(time.monotonic() - self._start, 4)

    def partial(self, text: str) -> None:
        self.events.append(["partial", self.now(), text])

    def final(self, text: str) -> None:
        self.events.append(["final", self.now(), text])
        if self._endpoint_at is not None:
            self._final.set()

    async def chunks(self) -> AsyncIterator[tuple[bytes, bool]]:
        """Yield (chunk, endpoint) at 1x speed.

        A chunk goes out once its last sample has been "spoken". `endpoint` is true for the
        chunk that completes the speech; the adapter sends its endpoint request right after
        it. After the clip, silence follows until the first final after the endpoint or the
        final timeout.
        """
        step = self.rate * 2 * self.chunk_ms // 1000
        self._start = time.monotonic()
        offset = index = 0
        while True:
            index += 1
            due = self._start + index * self.chunk_ms / 1000
            await asyncio.sleep(max(0.0, due - time.monotonic()))
            if offset < len(self.pcm):
                chunk = self.pcm[offset : offset + step]
                offset += len(chunk)
            elif (
                self._final.is_set()
                or self.now() - self._endpoint_at > self.final_timeout_s
            ):
                return
            else:
                chunk = bytes(step)
            endpoint = self._endpoint_at is None and (
                offset >= len(self.pcm) or offset / (2 * self.rate) >= self.speech_end_s
            )
            if endpoint:
                self._endpoint_at = self.now()
                self.events.append(["endpoint", self._endpoint_at, ""])
            yield chunk, endpoint

    def result(self) -> tuple[str, dict[str, Any]]:
        """Text known at the first final after the endpoint, and the latency figures."""
        end = self._endpoint_at
        after = [event for event in self.events if end is not None and event[1] >= end]
        first_final = next((t for kind, t, _ in after if kind == "final"), None)
        first_text = next(
            (
                t
                for kind, t, text in after
                if kind in {"partial", "final"} and text.strip()
            ),
            None,
        )
        cutoff = first_final if first_final is not None else float("inf")
        text = " ".join(
            text.strip()
            for kind, t, text in self.events
            if kind == "final" and t <= cutoff and text.strip()
        )
        return text, {
            "speech_end_s": self.speech_end_s,
            "endpoint_at_s": end,
            "final_after_endpoint": first_final is not None,
            "time_to_final_s": None
            if first_final is None
            else round(first_final - end, 4),
            "time_to_first_partial_s": None
            if first_text is None
            else round(first_text - end, 4),
        }


async def drive(
    ws: Any,
    stream: Stream,
    *,
    audio: Callable[[bytes], Awaitable[None]],
    handle: Callable[[str | bytes], bool | None],
    endpoint: Callable[[], Awaitable[None]] | None = None,
    close: Callable[[], Awaitable[None]] | None = None,
    drain_s: float = 5.0,
) -> None:
    """Send paced audio while handling messages; `handle` returns False to stop reading."""

    async def receive() -> None:
        async for message in ws:
            if handle(message) is False:
                return

    receiver = asyncio.create_task(receive())
    try:
        async for chunk, at_endpoint in stream.chunks():
            if receiver.done():
                receiver.result()  # surface a provider error or an early close
                raise ConnectionError("provider closed the stream early")
            await audio(chunk)
            if at_endpoint and endpoint is not None:
                await endpoint()
        if close is not None:
            await close()
        try:
            await asyncio.wait_for(asyncio.shield(receiver), timeout=drain_s)
        except (TimeoutError, websockets.ConnectionClosedError):
            pass  # the scored text is settled; a close without handshake changes nothing
    finally:
        receiver.cancel()


def _query(config) -> str:
    return urlencode(
        {
            name: str(value).lower() if isinstance(value, bool) else value
            for name, value in config.request.items()
        }
    )


async def _deepgram(config, key, stream: Stream) -> None:
    url = f"{config.connection['endpoint']}?model={config.model}&{_query(config)}"
    async with websockets.connect(
        url, additional_headers={"Authorization": f"Token {key}"}
    ) as ws:

        def handle(message):
            data = json.loads(message)
            if data.get("type") != "Results":
                return
            text = data["channel"]["alternatives"][0]["transcript"]
            (stream.final if data.get("is_final") else stream.partial)(text)

        await drive(
            ws,
            stream,
            audio=ws.send,
            handle=handle,
            endpoint=lambda: ws.send(json.dumps({"type": "Finalize"})),
            close=lambda: ws.send(json.dumps({"type": "CloseStream"})),
        )


async def _elevenlabs(config, key, stream: Stream) -> None:
    url = f"{config.connection['endpoint']}?model_id={config.model}&{_query(config)}"

    def chunk(pcm: bytes, commit: bool = False) -> str:
        return json.dumps(
            {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(pcm).decode(),
                "commit": commit,
                "sample_rate": stream.rate,
            }
        )

    async with websockets.connect(url, additional_headers={"xi-api-key": key}) as ws:

        def handle(message):
            data = json.loads(message)
            kind = data.get("message_type")
            if kind == "partial_transcript":
                stream.partial(data["text"])
            elif kind == "committed_transcript":
                stream.final(data["text"])
            elif "error" in data:  # every error type carries an `error` field
                raise RuntimeError(f"ElevenLabs {kind}: {data['error']}")

        await drive(
            ws,
            stream,
            audio=lambda pcm: ws.send(chunk(pcm)),
            handle=handle,
            endpoint=lambda: ws.send(chunk(b"", commit=True)),
            close=ws.close,  # no end-of-stream message; closing ends the drain
        )


async def _openai(config, key, stream: Stream) -> None:
    session = {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": stream.rate},
                "transcription": {"model": config.model, **config.request},
                "turn_detection": None,  # the runner commits the buffer at the endpoint
            }
        },
    }
    headers = {"Authorization": f"Bearer {key}"}
    async with websockets.connect(
        config.connection["endpoint"], additional_headers=headers
    ) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": session}))
        while (data := json.loads(await ws.recv()))["type"] != "session.updated":
            if data["type"] == "error":
                raise RuntimeError(f"OpenAI error: {data['error']}")
        texts: dict[str, str] = {}

        def handle(message):
            data = json.loads(message)
            kind = data["type"]
            if kind == "conversation.item.input_audio_transcription.delta":
                item = data["item_id"]
                texts[item] = texts.get(item, "") + data.get("delta", "")
                stream.partial(texts[item])
            elif kind == "conversation.item.input_audio_transcription.completed":
                stream.final(data["transcript"])
            elif kind in {
                "error",
                "conversation.item.input_audio_transcription.failed",
            }:
                raise RuntimeError(f"OpenAI {kind}: {data.get('error')}")

        await drive(
            ws,
            stream,
            audio=lambda pcm: ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm).decode(),
                    }
                )
            ),
            handle=handle,
            endpoint=lambda: ws.send(json.dumps({"type": "input_audio_buffer.commit"})),
            close=ws.close,  # no end-of-stream message; closing ends the drain
        )


async def _xai(config, key, stream: Stream) -> None:
    url = f"{config.connection['endpoint']}?model={config.model}&{_query(config)}"
    async with websockets.connect(
        url, additional_headers={"Authorization": f"Bearer {key}"}
    ) as ws:
        ready = json.loads(await ws.recv())  # audio must wait for transcript.created
        if ready.get("type") != "transcript.created":
            raise RuntimeError(f"xAI: {ready}")
        locked: list[str] = []  # chunk finals of the open utterance

        def handle(message):
            data = json.loads(message)
            kind = data.get("type")
            if kind == "error":
                raise RuntimeError(f"xAI error: {data.get('message')}")
            if kind == "transcript.done":
                return False  # the server drops the socket right after this
            if kind != "transcript.partial":
                return None
            # Chunk finals lock ~3 s pieces of the utterance; the utterance final
            # (speech_final) repeats them stitched into the whole utterance.
            if data["speech_final"]:
                locked.clear()
                stream.final(data["text"])
            elif data["is_final"]:
                locked.append(data["text"])
                stream.partial(" ".join(locked))
            else:
                stream.partial(" ".join([*locked, data["text"]]))
            return None

        await drive(
            ws,
            stream,
            audio=ws.send,
            handle=handle,
            endpoint=lambda: ws.send(json.dumps({"type": "finalize"})),
            close=lambda: ws.send(json.dumps({"type": "audio.done"})),
        )


async def _assemblyai(config, key, stream: Stream) -> None:
    url = (
        f"{config.connection['endpoint']}?speech_model={config.model}&{_query(config)}"
    )
    async with websockets.connect(url, additional_headers={"Authorization": key}) as ws:

        def handle(message):
            data = json.loads(message)
            if data.get("type") == "Error":
                raise RuntimeError(
                    f"AssemblyAI error {data.get('error_code')}: {data.get('error')}"
                )
            if data.get("type") == "Turn":
                # Each Turn message carries the whole turn so far; end_of_turn closes it.
                (stream.final if data.get("end_of_turn") else stream.partial)(
                    data["transcript"]
                )
            return False if data.get("type") == "Termination" else None

        await drive(
            ws,
            stream,
            audio=ws.send,
            handle=handle,
            endpoint=lambda: ws.send(json.dumps({"type": "ForceEndpoint"})),
            close=lambda: ws.send(json.dumps({"type": "Terminate"})),
        )


async def _soniox(config, key, stream: Stream) -> None:
    headers = {"Authorization": f"Bearer {key}"}
    async with websockets.connect(
        config.connection["endpoint"], additional_headers=headers
    ) as ws:
        await ws.send(json.dumps({"model": config.model, **config.request}))
        segment = ""

        def handle(message):
            nonlocal segment
            data = json.loads(message)
            if data.get("error_code"):
                raise RuntimeError(
                    f"Soniox error {data['error_code']}: {data.get('error_message')}"
                )
            # Final tokens arrive once each; a <fin> (finalize) or <end> token closes the segment.
            pending = ""
            for token in data.get("tokens", []):
                if token["text"] in {"<fin>", "<end>"}:
                    stream.final(segment)
                    segment = ""
                elif token.get("is_final"):
                    segment += token["text"]
                else:
                    pending += token["text"]
            if (segment + pending).strip():
                stream.partial(segment + pending)
            return False if data.get("finished") else None

        await drive(
            ws,
            stream,
            audio=ws.send,
            handle=handle,
            endpoint=lambda: ws.send(json.dumps({"type": "finalize"})),
            close=lambda: ws.send(""),  # an empty text frame ends the stream
        )


async def _cartesia(config, key, stream: Stream) -> None:
    url = f"{config.connection['endpoint']}?model={config.model}&{_query(config)}"
    async with websockets.connect(url, additional_headers={"X-API-Key": key}) as ws:
        segment = ""

        def handle(message):
            nonlocal segment
            data = json.loads(message)
            kind = data.get("type")
            if kind == "error":
                raise RuntimeError(
                    f"Cartesia error {data.get('error_code')}: {data.get('message')}"
                )
            # Finals are word-level deltas that must be concatenated verbatim (they can split a
            # word), so a segment is the finals up to the finalize ack (flush_done) or the end.
            if kind == "transcript" and data.get("is_final"):
                segment += data["text"]
                stream.partial(segment)
            elif kind == "transcript":
                stream.partial(segment + data["text"])
            elif kind in {"flush_done", "done"}:
                stream.final(segment)
                segment = ""
            return False if kind == "done" else None

        await drive(
            ws,
            stream,
            audio=ws.send,
            handle=handle,
            endpoint=lambda: ws.send("finalize"),
            close=lambda: ws.send("close"),
        )


async def _smallest(config, key, stream: Stream) -> None:
    url = f"{config.connection['endpoint']}?model={config.model}&{_query(config)}"
    async with websockets.connect(
        url, additional_headers={"Authorization": f"Bearer {key}"}
    ) as ws:

        def handle(message):
            data = json.loads(message)
            if data.get("type") == "error":
                raise RuntimeError(
                    f"Smallest error {data.get('error_code')}: {data.get('message')}"
                )
            if data.get("type") == "transcription":
                (stream.final if data.get("is_final") else stream.partial)(
                    data["transcript"]
                )
            return False if data.get("is_last") else None

        await drive(
            ws,
            stream,
            audio=ws.send,
            handle=handle,
            endpoint=lambda: ws.send(json.dumps({"type": "finalize"})),
            close=lambda: ws.send(json.dumps({"type": "close_stream"})),
        )


async def _reson8(config, key, stream: Stream) -> None:
    url = f"{config.connection['endpoint']}?{_query(config)}"
    async with websockets.connect(
        url, additional_headers={"Authorization": f"ApiKey {key}"}
    ) as ws:

        def handle(message):
            data = json.loads(message)
            if data.get("type") == "transcript":
                (stream.final if data.get("is_final") else stream.partial)(data["text"])

        # Errors arrive as close codes. There is no end-of-stream message: closing ends the session.
        await drive(
            ws,
            stream,
            audio=ws.send,
            handle=handle,
            endpoint=lambda: ws.send(json.dumps({"type": "flush_request"})),
            close=ws.close,
        )


async def _mistral(config, key, stream: Stream) -> None:
    url = f"{config.connection['endpoint']}?model={config.model}"
    async with websockets.connect(
        url, additional_headers={"Authorization": f"Bearer {key}"}
    ) as ws:
        # Like the SDK, configure only once the session exists.
        created = json.loads(await ws.recv())
        if created.get("type") != "session.created":
            raise RuntimeError(f"mistral session failed: {created}")
        audio_format = {"encoding": "pcm_s16le", "sample_rate": stream.rate}
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"audio_format": audio_format, **config.request},
                }
            )
        )
        delta, ended = [], False

        def handle(message):
            data = json.loads(message)
            kind = data.get("type")
            if kind == "error":
                raise RuntimeError(f"mistral error: {data['error']}")
            if kind == "transcription.text.delta":
                delta.append(data["text"])
                stream.partial("".join(delta))
            elif kind == "transcription.done":
                # Sent per flush (and at input_audio.end) with the text since the previous one.
                delta.clear()
                stream.final(data["text"])
                return not ended  # the server drops the socket after the last one
            return None

        async def close():
            nonlocal ended
            ended = True
            await ws.send(json.dumps({"type": "input_audio.end"}))

        await drive(
            ws,
            stream,
            audio=lambda chunk: ws.send(
                json.dumps(
                    {
                        "type": "input_audio.append",
                        "audio": base64.b64encode(chunk).decode(),
                    }
                )
            ),
            handle=handle,
            endpoint=lambda: ws.send(json.dumps({"type": "input_audio.flush"})),
            close=close,
        )


async def _meta(config, key, stream: Stream) -> None:
    async with websockets.connect(config.connection["endpoint"]) as ws:
        audio_encoding = {16000: "PCM_16KHZ", 24000: "PCM_24KHZ"}[stream.rate]
        handshake = {
            "authorization": {"accessToken": f"Bearer {key}"},
            "audioEncoding": audio_encoding,
        }
        await ws.send(
            json.dumps({**handshake, "model": config.model, **config.request})
        )
        ack = json.loads(await ws.recv())
        if "sessionId" not in ack:
            raise RuntimeError(f"meta handshake failed: {ack}")
        ended = False

        def handle(message):
            data = json.loads(message)
            if data.get("type") == "error":
                raise RuntimeError(f"meta error: {data.get('message')}")
            if data.get("type") == "transcript":
                (stream.final if data.get("final") else stream.partial)(
                    data["transcript"]
                )

        # PUSH_TO_TALK has no per-turn commit: endStream ends the turn and the input, so the
        # clip's tail after the endpoint is dropped, and the server closes after the final.
        async def audio(chunk):
            if not ended:
                await ws.send(chunk)

        async def endpoint():
            nonlocal ended
            ended = True
            await ws.send(json.dumps({"type": "endStream"}))

        async def messages():
            async for message in ws:
                yield message
            if ended:
                # Keep the receiver open while drive plays out the tail.
                await asyncio.Future()

        await drive(
            messages(), stream, audio=audio, handle=handle, endpoint=endpoint, drain_s=0
        )


async def _gemini(config, key, stream: Stream) -> None:
    headers = {"x-goog-api-key": key}
    async with websockets.connect(
        config.connection["endpoint"], additional_headers=headers
    ) as ws:
        setup = {
            "model": f"models/{config.model}",
            "generationConfig": {"responseModalities": ["TEXT"]},
            "inputAudioTranscription": config.request,
        }
        await ws.send(json.dumps({"setup": setup}))
        reply = json.loads(await ws.recv())
        if "setupComplete" not in reply:
            raise RuntimeError(f"gemini setup failed: {reply}")
        mime = f"audio/pcm;rate={stream.rate}"

        def handle(message):
            content = json.loads(message).get("serverContent") or {}
            if "interimInputTranscription" in content:
                stream.partial(content["interimInputTranscription"].get("text", ""))
            if "inputTranscription" in content:
                stream.final(content["inputTranscription"].get("text", ""))

        def audio(chunk):
            data = base64.b64encode(chunk).decode("ascii")
            return ws.send(
                json.dumps(
                    {"realtimeInput": {"audio": {"data": data, "mimeType": mime}}}
                )
            )

        # audioStreamEnd with automatic VAD on finalizes the turn at once ("hybrid VAD").
        await drive(
            ws,
            stream,
            audio=audio,
            handle=handle,
            endpoint=lambda: ws.send(
                json.dumps({"realtimeInput": {"audioStreamEnd": True}})
            ),
        )


async def _chirp(config, _key, stream: Stream) -> None:
    # Speech-to-Text V2 streams only over gRPC and takes OAuth credentials (ADC), not API
    # keys. It has no finalize request: half-closing the stream ends it, so finals before
    # that come from the model's own endpointing.
    import google.auth
    from google.api_core.client_options import ClientOptions
    from google.cloud import speech_v2

    credentials, adc_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or adc_project
    location = config.connection["location"]
    request = config.request
    setup = speech_v2.StreamingRecognizeRequest(
        recognizer=f"projects/{project}/locations/{location}/recognizers/_",
        streaming_config={
            "config": {
                "explicit_decoding_config": {
                    "encoding": "LINEAR16",
                    "sample_rate_hertz": stream.rate,
                    "audio_channel_count": 1,
                },
                "language_codes": [request["language_code"]],
                "model": config.model,
                "features": {
                    "enable_automatic_punctuation": request[
                        "enable_automatic_punctuation"
                    ]
                },
            },
            "streaming_features": {"interim_results": request["interim_results"]},
        },
    )
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def requests():
        yield setup
        while (chunk := await queue.get()) is not None:
            yield speech_v2.StreamingRecognizeRequest(audio=chunk)

    def handle(response):
        for result in response.results:
            text = result.alternatives[0].transcript if result.alternatives else ""
            (stream.final if result.is_final else stream.partial)(text)

    options = ClientOptions(api_endpoint=config.connection["endpoint"])
    async with speech_v2.SpeechAsyncClient(
        credentials=credentials, client_options=options
    ) as client:
        responses = await client.streaming_recognize(requests=requests())
        await drive(
            responses,
            stream,
            audio=queue.put,
            handle=handle,
            close=lambda: queue.put(None),
        )


async def _microsoft(config, key, stream: Stream) -> None:
    host = os.environ[config.connection["endpoint_env"]].split("://")[-1].rstrip("/")
    url = f"wss://{host}{config.connection['path']}?intent=transcription"
    async with websockets.connect(url, additional_headers={"api-key": key}) as ws:
        session = {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": stream.rate},
                    "transcription": {"model": config.model, **config.request},
                    "turn_detection": None,
                    "noise_reduction": None,
                }
            },
        }
        await ws.send(json.dumps({"type": "session.update", "session": session}))
        while (event := json.loads(await ws.recv()))["type"] != "session.updated":
            if event["type"] == "error":
                raise RuntimeError(f"microsoft error: {event.get('error')}")
        done = ""  # finalized `delta` text of the open commit interval

        def handle(message):
            nonlocal done
            event = json.loads(message)
            kind = event["type"].removeprefix(
                "conversation.item.input_audio_transcription."
            )
            if kind == "delta":
                done += event["delta"]
                stream.partial(done)
            elif kind == "intermediate":
                stream.partial(done + event["intermediate"])
            elif kind == "completed":
                # The whole transcript since the previous commit: one segment per commit.
                stream.final(event["transcript"])
                done = ""
            elif kind in {"error", "failed"}:
                raise RuntimeError(f"microsoft error: {event.get('error', event)}")

        def audio(chunk):
            data = base64.b64encode(chunk).decode("ascii")
            return ws.send(
                json.dumps({"type": "input_audio_buffer.append", "audio": data})
            )

        await drive(
            ws,
            stream,
            audio=audio,
            handle=handle,
            endpoint=lambda: ws.send(json.dumps({"type": "input_audio_buffer.commit"})),
        )


async def _together(config, key, stream: Stream) -> None:
    url = f"{config.connection['endpoint']}?model={config.model}&{_query(config)}"
    async with websockets.connect(
        url, additional_headers={"Authorization": f"Bearer {key}"}
    ) as ws:
        while (event := json.loads(await ws.recv()))["type"] != "session.created":
            if event["type"] == "error":
                raise RuntimeError(f"together error: {event.get('error', event)}")

        def handle(message):
            event = json.loads(message)
            kind = event["type"].removeprefix(
                "conversation.item.input_audio_transcription."
            )
            if kind == "delta":  # the whole current hypothesis of the open segment
                stream.partial(event.get("delta", ""))
            elif kind == "completed":
                stream.final(event.get("transcript", ""))
            elif kind in {"error", "failed"}:
                raise RuntimeError(f"together error: {event.get('error', event)}")

        def audio(chunk):
            data = base64.b64encode(chunk).decode("ascii")
            return ws.send(
                json.dumps({"type": "input_audio_buffer.append", "audio": data})
            )

        await drive(
            ws,
            stream,
            audio=audio,
            handle=handle,
            endpoint=lambda: ws.send(json.dumps({"type": "input_audio_buffer.commit"})),
        )


async def _deepgram_flux(config, key, stream: Stream) -> None:
    url = f"{config.connection['endpoint']}?model={config.model}&{_query(config)}"
    async with websockets.connect(
        url, additional_headers={"Authorization": f"Token {key}"}
    ) as ws:

        def handle(message):
            data = json.loads(message)
            if data.get("type") == "Error":
                raise RuntimeError(
                    f"Deepgram Flux error {data.get('code')}: {data.get('description')}"
                )
            if data.get("type") == "TurnInfo":
                # Each TurnInfo carries the whole turn so far; EndOfTurn closes it.
                (stream.final if data["event"] == "EndOfTurn" else stream.partial)(
                    data["transcript"]
                )

        # eot_threshold=1.0 leaves turn endings to ForceEndTurn (plus the eot_timeout_ms
        # silence backstop). CloseStream does not end the open turn.
        await drive(
            ws,
            stream,
            audio=ws.send,
            handle=handle,
            endpoint=lambda: ws.send(json.dumps({"type": "ForceEndTurn"})),
            close=lambda: ws.send(json.dumps({"type": "CloseStream"})),
        )


async def _speechmatics(config, key, stream: Stream) -> None:
    headers = {"Authorization": f"Bearer {key}"}
    async with websockets.connect(
        config.connection["endpoint"], additional_headers=headers
    ) as ws:
        start = {
            "message": "StartRecognition",
            "audio_format": {
                "type": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": stream.rate,
            },
            "transcription_config": {"model": config.model, **config.request},
            # ForceEndOfUtterance works only in external mode, where nothing else ends a turn.
            "turn_config": {"turn_detection_mode": "external"},
        }
        await ws.send(json.dumps(start))
        while (data := json.loads(await ws.recv()))["message"] != "RecognitionStarted":
            if data["message"] == "Error":
                raise RuntimeError(f"Speechmatics {data['type']}: {data['reason']}")
        sent = chunks = 0  # audio bytes and AddAudio messages so far
        segments: list[str] = []  # final segments of the open turn

        def handle(message):
            data = json.loads(message)
            kind = data["message"]
            # A turn can close several segments; EndOfTurn comes after the last one.
            if kind == "AddPartialSegment":
                stream.partial(" ".join([*segments, data["segment"]["transcript"]]))
            elif kind == "AddSegment":
                segments.append(data["segment"]["transcript"])
            elif kind == "EndOfTurn":
                stream.final(" ".join(segments))
                segments.clear()
            elif kind == "Error":
                raise RuntimeError(f"Speechmatics {data['type']}: {data['reason']}")
            return False if kind == "EndOfTranscript" else None

        async def audio(chunk):
            nonlocal sent, chunks
            sent, chunks = sent + len(chunk), chunks + 1
            await ws.send(chunk)

        def endpoint():
            # Cut the turn at the audio sent so far, not where the message lands.
            seconds = sent / (2 * stream.rate)
            return ws.send(
                json.dumps({"message": "ForceEndOfUtterance", "timestamp": seconds})
            )

        await drive(
            ws,
            stream,
            audio=audio,
            handle=handle,
            endpoint=endpoint,
            close=lambda: ws.send(
                json.dumps({"message": "EndOfStream", "last_seq_no": chunks})
            ),
        )


async def _azure_speech(config, key, stream: Stream) -> None:
    # Speech SDK continuous recognition from a push stream. The SDK runs on its own
    # threads; results reach the loop through a queue. PushAudioInputStream.commit (SDK
    # 1.52, preview, needs setfeature=forcecommit) finalizes the open segment at the
    # endpoint and keeps the stream open. The service's silence segmentation stays on.
    import azure.cognitiveservices.speech as speechsdk

    request = config.request
    speech_config = speechsdk.SpeechConfig(
        subscription=key, endpoint=os.environ[config.connection["endpoint_env"]]
    )
    speech_config.speech_recognition_language = request["language"]
    speech_config.set_profanity(speechsdk.ProfanityOption[request["profanity"]])
    speech_config.set_service_property(
        "setfeature",
        request["setfeature"],
        speechsdk.ServicePropertyChannel.UriQueryParameter,
    )
    audio_format = speechsdk.audio.AudioStreamFormat(
        samples_per_second=stream.rate, bits_per_sample=16, channels=1
    )
    push = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=speechsdk.audio.AudioConfig(stream=push),
    )
    loop = asyncio.get_running_loop()
    results: asyncio.Queue = asyncio.Queue()

    def put(result) -> None:
        # A result reads all its fields on creation, so it can cross threads.
        loop.call_soon_threadsafe(results.put_nowait, result)

    for signal in (recognizer.recognizing, recognizer.recognized, recognizer.canceled):
        signal.connect(lambda event: put(event.result))
    connection = speechsdk.Connection.from_recognizer(recognizer)
    connection.connected.connect(lambda _: put(None))

    def handle(result):
        reason = None if result is None else result.reason
        if reason == speechsdk.ResultReason.RecognizingSpeech:
            stream.partial(result.text)
        elif reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            if details.reason == speechsdk.CancellationReason.Error:
                raise RuntimeError(
                    f"Azure STT {details.code.name}: {details.error_details}"
                )
            return False  # EndOfStream: the session is over after `close`
        elif reason == speechsdk.ResultReason.RecognizedSpeech or (
            result is not None and result.commit_token
        ):
            # A NoMatch result acknowledges a commit whose segment has no speech.
            stream.final(result.text)
        return None

    async def messages():
        while True:
            yield await results.get()

    async def audio(chunk):
        push.write(chunk)

    async def endpoint():
        if not push.commit():
            raise RuntimeError("Azure STT rejected the commit")

    async def close():
        push.close()  # end of audio: the service ends the session

    try:
        # Connect before the clock starts, as the websocket adapters do.
        connection.open(True)
        handle(await results.get())  # None once connected; raises on failure
        await asyncio.to_thread(
            lambda: recognizer.start_continuous_recognition_async().get()
        )
        await drive(
            messages(),
            stream,
            audio=audio,
            handle=handle,
            endpoint=endpoint,
            close=close,
        )
    finally:
        await asyncio.to_thread(
            lambda: recognizer.stop_continuous_recognition_async().get()
        )


async def _nari(config, key, stream: Stream) -> None:
    async with websockets.connect(
        config.connection["endpoint"],
        additional_headers={"Authorization": f"Bearer {key}"},
    ) as ws:
        session = {"model": config.model, **config.request}
        await ws.send(json.dumps({"type": "session.configure", "session": session}))
        while (event := json.loads(await ws.recv()))["type"] != "session.configured":
            if event["type"] == "error":
                raise RuntimeError(f"Nari error: {event.get('error')}")

        def handle(message):
            data = json.loads(message)
            kind = data.get("type")
            if kind == "error":
                raise RuntimeError(f"Nari error: {data.get('error')}")
            # Each partial is the whole current hypothesis of its utterance.
            if kind == "transcript.partial":
                stream.partial(data["transcript"])
            elif kind == "transcript.completed":
                stream.final(data["transcript"])

        def audio(chunk):
            data = base64.b64encode(chunk).decode("ascii")
            return ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": data}))

        # Manual commit (no turn_detection): the endpoint's commit finalizes the utterance.
        await drive(
            ws,
            stream,
            audio=audio,
            handle=handle,
            endpoint=lambda: ws.send(json.dumps({"type": "input_audio_buffer.commit"})),
        )


ADAPTERS: dict[str, Callable[..., Awaitable[None]]] = {
    "deepgram_realtime": _deepgram,
    "elevenlabs_realtime": _elevenlabs,
    "openai_realtime": _openai,
    "xai_realtime": _xai,
    "assemblyai_realtime": _assemblyai,
    "soniox_realtime": _soniox,
    "cartesia_realtime": _cartesia,
    "smallest_realtime": _smallest,
    "reson8_realtime": _reson8,
    "mistral_realtime": _mistral,
    "meta_realtime": _meta,
    "gemini_realtime": _gemini,
    "google_chirp3_realtime": _chirp,
    "microsoft_realtime": _microsoft,
    "together_realtime": _together,
    "deepgram_flux_realtime": _deepgram_flux,
    "speechmatics_realtime": _speechmatics,
    "azure_speech_realtime": _azure_speech,
    "nari_realtime": _nari,
}
