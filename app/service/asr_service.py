import tempfile
from pathlib import Path

from app.api.schemas import AsrResponse
from app.domain import FileItem, ModelBundle
from funasr import AutoModel

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
    suffix = Path(file_item.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_item.data)
        tmp_path = tmp.name

    try:
        return transcribe_from_path(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def transcribe_from_path(wav_path: str) -> AsrResponse:
    bundle = _get_bundle()
    res = bundle.funasr_model.generate(input=[wav_path], cache={}, batch_size_s=300, batch_size_threshold_s=60)
    return AsrResponse(text=res[0]["text"])


def get_stream_and_offline_models() -> tuple[AutoModel, AutoModel]:
    bundle = _get_bundle()
    return bundle.streaming_funasr_model, bundle.funasr_model
