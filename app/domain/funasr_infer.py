from pathlib import Path
import tempfile

from app.api.schemas import AsrResponse
from app.domain.file_item import FileItem
from app.domain.funasr_loader import ModelBundle


def infer_from_file_item(
    bundle: ModelBundle,
    file_item: FileItem,
) -> AsrResponse:
    suffix = Path(file_item.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_item.data)
        tmp_path = tmp.name

    try:
        return infer_from_path(bundle, tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def infer_from_path(
    bundle: ModelBundle,
    wav_path: str,
) -> AsrResponse:
    text = _transcribe_funasr_path(bundle, wav_path)
    return AsrResponse(text=text)


def _transcribe_funasr_path(bundle: ModelBundle, wav_path: str) -> str:
    res = bundle.funasr_model.generate(input=[wav_path], cache={}, batch_size_s=300, batch_size_threshold_s=60)
    return res[0]["text"]
