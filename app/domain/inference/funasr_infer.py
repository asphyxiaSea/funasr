from app.domain.loaders.funasr_loader import ModelBundle
from app.domain.schemas.asr import AsrResponse, FileItem, Mode, MODE_VAD


def infer_from_file_item(
    bundle: ModelBundle,
    file_item: FileItem,
    mode: Mode,
) -> AsrResponse:
    if mode == MODE_VAD:
        text = _transcribe_funasr_bytes(bundle, file_item.data)
    else:
        text = _transcribe_direct_bytes(bundle, file_item.data)
    return AsrResponse(text=text, mode=mode)


def infer_from_path(
    bundle: ModelBundle,
    wav_path: str,
    mode: Mode,
) -> AsrResponse:
    if mode == MODE_VAD:
        text = _transcribe_funasr_path(bundle, wav_path)
    else:
        text = _transcribe_direct_path(bundle, wav_path)
    return AsrResponse(text=text, mode=mode)


def _transcribe_direct_bytes(bundle: ModelBundle, data: bytes) -> str:
    res = bundle.direct_model.inference(data_in=[data], **bundle.direct_kwargs)
    return res[0][0]["text"]


def _transcribe_funasr_bytes(bundle: ModelBundle, data: bytes) -> str:
    res = bundle.funasr_model.generate(input=[data], cache={}, batch_size=1)
    return res[0]["text"]


def _transcribe_direct_path(bundle: ModelBundle, wav_path: str) -> str:
    res = bundle.direct_model.inference(data_in=[wav_path], **bundle.direct_kwargs)
    return res[0][0]["text"]


def _transcribe_funasr_path(bundle: ModelBundle, wav_path: str) -> str:
    res = bundle.funasr_model.generate(input=[wav_path], cache={}, batch_size=1)
    return res[0]["text"]
