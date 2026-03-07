from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # 项目path
    base_dir: Path
    # funasr_asr模型
    funasr_asr_model_dir: str
    # funasr_streaming_asr模型
    funasr_stream_asr_model_dir: str
    # funasr_vad模型
    funasr_vad_model_dir: str
    # funasr_punc模型
    funasr_punc_model_dir: str
    # funasr_spk模型
    funasr_spk_model_dir: str
    device: str


@lru_cache
def get_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[2]

    return Settings(
        base_dir=base_dir,
        funasr_asr_model_dir=os.getenv(
            "FUNASR_ASR_MODEL_DIR",
            "models/ASRmodels/paraformer-zh",
        ),
        funasr_stream_asr_model_dir=os.getenv(
            "FUNASR_STREAM_ASR_MODEL_DIR",
            "models/ASRmodels/paraformer-zh-streaming",
        ),
        funasr_vad_model_dir=os.getenv(
            "FUNASR_VAD_MODEL_DIR",
            "models/VADmodels/speech_fsmn_vad_zh",
        ),
        funasr_punc_model_dir=os.getenv(
            "FUNASR_PUNC_MODEL_DIR",
            "models/PUNCmodels/ct-punc",
        ),
        funasr_spk_model_dir=os.getenv(
            "FUNASR_SPK_MODEL_DIR",
            "models/SPKmodels/cam++",
        ),
        device=os.getenv("ASR_DEVICE", "cuda:1"),
    )
