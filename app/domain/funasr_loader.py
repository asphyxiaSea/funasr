from dataclasses import dataclass

from funasr import AutoModel

from app.config.settings import Settings


@dataclass(frozen=True)
class ModelBundle:
    funasr_model: AutoModel
    streaming_funasr_model: AutoModel


def load_models(settings: Settings) -> ModelBundle:
    funasr_model = AutoModel(
        model=settings.funasr_asr_model_dir,
        vad_model=settings.funasr_vad_model_dir,
        vad_kwargs={"max_single_segment_time": 30000},
        punc_model=settings.funasr_punc_model_dir,
        spk_model=settings.funasr_spk_model_dir,
        device=settings.device,
    )

    streaming_funasr_model = AutoModel(
        model=settings.funasr_stream_asr_model_dir,
        device=settings.device,
    )

    return ModelBundle(
        funasr_model=funasr_model,
        streaming_funasr_model=streaming_funasr_model,
    )
