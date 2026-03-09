from dataclasses import dataclass

from funasr import AutoModel

from app.config.settings import Settings


@dataclass(frozen=True)
class ModelBundle:
    offline_model: AutoModel
    stream_model: AutoModel
    spk_model: AutoModel


_bundle: ModelBundle | None = None
_init_error: str | None = None


def initialize_models(settings: Settings) -> None:
    global _bundle, _init_error
    if _bundle is not None:
        return

    try:
        stream_model = AutoModel(
            model=settings.stream_asr_model_dir,
            disable_update=True,
            device=settings.device,
        )
        offline_model = AutoModel(
            model=settings.asr_model_dir,
            vad_model=settings.vad_model_dir,
            punc_model=settings.punc_model_dir,
            spk_model=settings.spk_model_dir,
            disable_update=True,
            device=settings.device,
        )
        spk_model = AutoModel(
            model=settings.spk_model_dir,
            disable_update=True,
            device=settings.device,
        )
        _bundle = ModelBundle(
            offline_model=offline_model,
            stream_model=stream_model,
            spk_model=spk_model,
        )
        _init_error = None
    except Exception as exc:
        _bundle = None
        _init_error = str(exc)
        raise


def get_models() -> ModelBundle:
    if _bundle is None:
        raise RuntimeError(_init_error or "Models not initialized")
    return _bundle


def is_ready() -> bool:
    return _bundle is not None


def get_init_error() -> str | None:
    return _init_error
