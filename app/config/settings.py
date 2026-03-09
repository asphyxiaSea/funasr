import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    asr_model_dir: str
    stream_asr_model_dir: str
    vad_model_dir: str
    punc_model_dir: str
    spk_model_dir: str
    chunk_size: tuple[int, int, int]
    encoder_chunk_look_back: int
    decoder_chunk_look_back: int
    sample_rate: int

    @property
    def chunk_stride(self) -> int:
        # Keep text3.py-compatible stride semantics (10 * 960 samples).
        return self.chunk_size[1] * 960


@lru_cache
def get_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[2]

    return Settings(
        base_dir=base_dir,
        asr_model_dir=os.getenv("FUNASR_ASR_MODEL_DIR", "models/ASRmodels/paraformer-zh"),
        stream_asr_model_dir=os.getenv(
            "FUNASR_STREAM_ASR_MODEL_DIR", "models/ASRmodels/paraformer-zh-streaming"
        ),
        vad_model_dir=os.getenv("FUNASR_VAD_MODEL_DIR", "models/VADmodels/speech_fsmn_vad_zh"),
        punc_model_dir=os.getenv("FUNASR_PUNC_MODEL_DIR", "models/PUNCmodels/ct-punc"),
        spk_model_dir=os.getenv("FUNASR_SPK_MODEL_DIR", "models/SPKmodels/cam++"),
        chunk_size=(0, 10, 5),
        encoder_chunk_look_back=4,
        decoder_chunk_look_back=1,
        sample_rate=16000,
    )
