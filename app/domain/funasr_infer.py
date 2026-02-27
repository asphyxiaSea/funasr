from pathlib import Path
import tempfile

from app.domain.asr import AsrResponse, FileItem, Mode, MODE_FUNASR
from app.domain.funasr_loader import ModelBundle


def infer_from_file_item(
    bundle: ModelBundle,
    file_item: FileItem,
    mode: Mode,
) -> AsrResponse:
    suffix = Path(file_item.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_item.data)
        tmp_path = tmp.name

    try:
        return infer_from_path(bundle, tmp_path, mode)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def infer_from_path(
    bundle: ModelBundle,
    wav_path: str,
    mode: Mode,
) -> AsrResponse:
    if mode == MODE_FUNASR:
        text = _transcribe_funasr_path(bundle, wav_path)
    else:
        text = _transcribe_direct_path(bundle, wav_path)
    return AsrResponse(text=text, mode=mode)


def _transcribe_direct_path(bundle: ModelBundle, wav_path: str) -> str:
    res = bundle.direct_model.inference(data_in=[wav_path], **bundle.direct_kwargs)
    return res[0][0]["text"]


def _transcribe_funasr_path(bundle: ModelBundle, wav_path: str) -> str:
    res = bundle.funasr_model.generate(input=[wav_path], cache={}, batch_size_s=300, batch_size_threshold_s=60)
    return res[0]["text"]
