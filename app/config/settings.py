from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # 项目path
    base_dir: Path
    # 源码path
    funasr_dir: Path
    # 直接推理asr模型位置
    direct_asr_model_dir: str
    # funasr_asr模型
    funasr_asr_model_dir: str
    # funasr_vad模型
    funasr_vad_model_dir: str
    device: str


@lru_cache
def get_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[2]
    funasr_dir = Path(os.getenv("FUNASR_DIR", str(base_dir / "FunASR")))

    return Settings(
        base_dir=base_dir,
        funasr_dir=funasr_dir,
        direct_asr_model_dir=os.getenv(
            "DIRECT_ASR_MODEL_DIR",
            "models/ASRmodels/Fun-ASR-Nano-2512",
        ),
        funasr_asr_model_dir=os.getenv(
            "FUNASR_ASR_MODEL_DIR",
            "models/ASRmodels/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        ),
        funasr_vad_model_dir=os.getenv(
            "FUNASR_VAD_MODEL_DIR",
            "models/VADmodels/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        ),
        device=os.getenv("ASR_DEVICE", "cuda:1"),
    )
