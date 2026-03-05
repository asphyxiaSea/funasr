from pathlib import Path
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.api.schemas import AsrResponse, MODE_FUNASR
from app.domain.file_item import FileItem
from app.domain.funasr_loader import ModelBundle


@dataclass
class Pcm16StreamSession:
    bundle: ModelBundle
    sample_rate: int
    cache: dict[str, Any] = field(default_factory=dict)
    pending_bytes: bytes = b""

    def feed(self, chunk: bytes) -> str:
        if self.bundle.streaming_funasr_model is None:
            raise RuntimeError("streaming model not initialized")

        self.pending_bytes += chunk
        frame_size = 2
        aligned_size = (len(self.pending_bytes) // frame_size) * frame_size
        if aligned_size == 0:
            return ""

        pcm_bytes = self.pending_bytes[:aligned_size]
        self.pending_bytes = self.pending_bytes[aligned_size:]

        pcm_float = _pcm16_to_float32(pcm_bytes)
        if pcm_float.size == 0:
            return ""
        return _transcribe_stream_chunk(
            bundle=self.bundle,
            pcm_float=pcm_float,
            cache=self.cache,
            is_final=False,
            sample_rate=self.sample_rate,
        )

    def finalize(self) -> str:
        if self.bundle.streaming_funasr_model is None:
            raise RuntimeError("streaming model not initialized")

        tail = ""
        if self.pending_bytes:
            aligned_size = (len(self.pending_bytes) // 2) * 2
            if aligned_size > 0:
                tail_pcm = _pcm16_to_float32(self.pending_bytes[:aligned_size])
                self.pending_bytes = self.pending_bytes[aligned_size:]
                if tail_pcm.size > 0:
                    tail = _transcribe_stream_chunk(
                        bundle=self.bundle,
                        pcm_float=tail_pcm,
                        cache=self.cache,
                        is_final=False,
                        sample_rate=self.sample_rate,
                    )

        final_text = _transcribe_stream_chunk(
            bundle=self.bundle,
            pcm_float=np.array([], dtype=np.float32),
            cache=self.cache,
            is_final=True,
            sample_rate=self.sample_rate,
        )
        if final_text:
            return final_text
        return tail


def create_pcm16_stream_session(bundle: ModelBundle, sample_rate: int) -> Pcm16StreamSession:
    return Pcm16StreamSession(bundle=bundle, sample_rate=sample_rate)


def infer_from_file_item(
    bundle: ModelBundle,
    file_item: FileItem,
) -> AsrResponse:
    suffix = Path(file_item.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_item.data)
        tmp_path = tmp.name

    try:
        return infer_from_path(bundle, tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def infer_from_path(
    bundle: ModelBundle,
    wav_path: str,
) -> AsrResponse:
    text = _transcribe_funasr_path(bundle, wav_path)
    return AsrResponse(text=text, mode=MODE_FUNASR)


def _transcribe_funasr_path(bundle: ModelBundle, wav_path: str) -> str:
    res = bundle.funasr_model.generate(input=[wav_path], cache={}, batch_size_s=300, batch_size_threshold_s=60)
    return res[0]["text"]


def _pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return (pcm_int16.astype(np.float32)) / 32768.0


def _transcribe_stream_chunk(
    bundle: ModelBundle,
    pcm_float: np.ndarray,
    cache: dict[str, Any],
    is_final: bool,
    sample_rate: int,
) -> str:
    if bundle.streaming_funasr_model is None:
        raise RuntimeError("streaming model not initialized")

    res = bundle.streaming_funasr_model.generate(
        input=pcm_float,
        cache=cache,
        is_final=is_final,
        sample_rate=sample_rate,
        **bundle.streaming_kwargs,
    )
    return _extract_text(res)


def _extract_text(result: Any) -> str:
    if isinstance(result, list) and result:
        item = result[0]
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                return text
    return ""
