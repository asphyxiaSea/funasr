from dataclasses import dataclass
from typing import Dict

from funasr import AutoModel

from app.config.settings import Settings


@dataclass(frozen=True)
class ModelBundle:
    funasr_model: AutoModel
    streaming_funasr_model: AutoModel | None
    streaming_kwargs: Dict[str, object]


def load_models(settings: Settings) -> ModelBundle:
    funasr_model = AutoModel(
        model=settings.funasr_asr_model_dir,
        vad_model=settings.funasr_vad_model_dir,
        vad_kwargs={"max_single_segment_time": 30000},
        punc_model=settings.funasr_punc_model_dir,
        spk_model=settings.funasr_spk_model_dir,
        device=settings.device,
        disable_update=True,
    )

    streaming_funasr_model: AutoModel | None = None
    if settings.funasr_stream_asr_model_dir:
        streaming_funasr_model = AutoModel(
            model=settings.funasr_stream_asr_model_dir,
            device=settings.device,
            disable_update=True,
        )

    streaming_kwargs: Dict[str, object] = {
        "chunk_size": settings.streaming_chunk_size,
        "encoder_chunk_look_back": settings.streaming_encoder_chunk_look_back,
        "decoder_chunk_look_back": settings.streaming_decoder_chunk_look_back,
    }

    return ModelBundle(
        funasr_model=funasr_model,
        streaming_funasr_model=streaming_funasr_model,
        streaming_kwargs=streaming_kwargs,
    )
