from app.domain.inference.funasr_infer import infer_from_file_item, infer_from_path
from app.domain.loaders.funasr_loader import ModelBundle
from app.domain.schemas.asr import AsrResponse, FileItem, Mode

_model_bundle: ModelBundle | None = None


def init_models(bundle: ModelBundle) -> None:
    global _model_bundle
    _model_bundle = bundle


def _get_bundle() -> ModelBundle:
    if _model_bundle is None:
        raise RuntimeError("Models not initialized")
    return _model_bundle


def transcribe_from_file_item(file_item: FileItem, mode: Mode) -> AsrResponse:
    return infer_from_file_item(_get_bundle(), file_item, mode)


def transcribe_from_path(wav_path: str, mode: Mode) -> AsrResponse:
    return infer_from_path(_get_bundle(), wav_path, mode)
