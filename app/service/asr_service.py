from app.api.schemas import AsrResponse
from app.domain import FileItem, ModelBundle
from app.domain import create_pcm16_stream_session as create_domain_pcm16_stream_session
from app.domain import infer_from_file_item, infer_from_path
from app.domain.funasr_infer import Pcm16StreamSession

_model_bundle: ModelBundle | None = None
_init_error: str | None = None


def init_models(bundle: ModelBundle) -> None:
    global _model_bundle, _init_error
    _model_bundle = bundle
    _init_error = None


def set_init_error(message: str) -> None:
    global _model_bundle, _init_error
    _model_bundle = None
    _init_error = message


def get_model_status() -> tuple[bool, str | None]:
    if _model_bundle is not None:
        return True, None
    return False, _init_error or "Models not initialized"


def _get_bundle() -> ModelBundle:
    if _model_bundle is None:
        raise RuntimeError("Models not initialized")
    return _model_bundle


def transcribe_from_file_item(file_item: FileItem) -> AsrResponse:
    return infer_from_file_item(_get_bundle(), file_item)


def transcribe_from_path(wav_path: str) -> AsrResponse:
    return infer_from_path(_get_bundle(), wav_path)


def create_pcm16_stream_session(sample_rate: int) -> Pcm16StreamSession:
    return create_domain_pcm16_stream_session(_get_bundle(), sample_rate=sample_rate)
