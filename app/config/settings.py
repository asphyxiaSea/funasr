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
    # 流式采样率
    streaming_sample_rate: int
    # 流式声道数
    streaming_channels: int
    # 流式分片参数
    streaming_chunk_size: list[int]
    # 流式encoder回看
    streaming_encoder_chunk_look_back: int
    # 流式decoder回看
    streaming_decoder_chunk_look_back: int
    device: str


def _parse_chunk_size(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


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
        streaming_sample_rate=int(os.getenv("STREAMING_SAMPLE_RATE", "16000")),
        streaming_channels=int(os.getenv("STREAMING_CHANNELS", "1")),
        streaming_chunk_size=_parse_chunk_size(os.getenv("STREAMING_CHUNK_SIZE", "0,10,5")),
        streaming_encoder_chunk_look_back=int(os.getenv("STREAMING_ENCODER_CHUNK_LOOK_BACK", "4")),
        streaming_decoder_chunk_look_back=int(os.getenv("STREAMING_DECODER_CHUNK_LOOK_BACK", "1")),
        device=os.getenv("ASR_DEVICE", "cuda:1"),
    )
