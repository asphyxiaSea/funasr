import tempfile
from pathlib import Path

import numpy as np

from app.config.settings import Settings
from app.domain.models import get_models
from app.domain.streaming import StreamSession


def _pcm16_to_float32(pcm_bytes: bytes | bytearray) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_path(wav_path: str) -> str:
    models = get_models()
    result = models.offline_model.generate(input=wav_path)
    return result[0]["text"] if result else ""


def transcribe_upload(file_bytes: bytes, filename: str) -> str:
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        return transcribe_path(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def process_stream_bytes(session: StreamSession, chunk_bytes: bytes, settings: Settings) -> list[str]:
    models = get_models()
    session.append_chunk(chunk_bytes)

    frame_bytes = settings.chunk_stride * 2
    partials: list[str] = []

    while True:
        frame = session.pop_stream_frame(frame_bytes)
        if frame is None:
            break

        speech_chunk = _pcm16_to_float32(frame)
        result = models.stream_model.generate(
            input=speech_chunk,
            cache=session.cache,
            is_final=False,
            chunk_size=list(settings.chunk_size),
            encoder_chunk_look_back=settings.encoder_chunk_look_back,
            decoder_chunk_look_back=settings.decoder_chunk_look_back,
        )
        if result and result[0].get("text"):
            partials.append(result[0]["text"])

    return partials


def finalize_stream(session: StreamSession, settings: Settings) -> str:
    models = get_models()

    if session.audio_buffer:
        speech_chunk = _pcm16_to_float32(session.audio_buffer)
        result = models.stream_model.generate(
            input=speech_chunk,
            cache=session.cache,
            is_final=True,
            chunk_size=list(settings.chunk_size),
            encoder_chunk_look_back=settings.encoder_chunk_look_back,
            decoder_chunk_look_back=settings.decoder_chunk_look_back,
        )
        session.audio_buffer.clear()
    else:
        result = models.stream_model.generate(
            input=None,
            cache=session.cache,
            is_final=True,
            chunk_size=list(settings.chunk_size),
            encoder_chunk_look_back=settings.encoder_chunk_look_back,
            decoder_chunk_look_back=settings.decoder_chunk_look_back,
        )

    return result[0]["text"] if result else ""


def rerun_full_audio(session: StreamSession):
    models = get_models()
    full_audio = _pcm16_to_float32(session.full_audio_buffer)
    result = models.offline_model.generate(input=full_audio)
    return result
