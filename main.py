from typing import Any, Dict, Literal, cast
from pathlib import Path
import sys
import tempfile

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from funasr import AutoModel

BASE_DIR = Path(__file__).resolve().parent
FUNASR_DIR = BASE_DIR / "FunASR"   # 根据你的实际目录调整

sys.path.append(str(FUNASR_DIR))

from FunASR.model import FunASRNano

app = FastAPI(title="FunASR Service")

MODE_DIRECT: Literal["direct"] = "direct"
MODE_VAD: Literal["vad"] = "vad"
Mode = Literal["direct", "vad"]

DIRECT_MODEL_DIR = "models/ASRmodels/Fun-ASR-Nano-2512"
ASR_VAD_MODEL_DIR = (
    "models/ASRmodels/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)
VAD_MODEL_DIR = "models/VADmodels/speech_fsmn_vad_zh-cn-16k-common-pytorch"

_direct_model_cache: tuple[FunASRNano, Dict[str, Any]] | None = None
_vad_model_cache: AutoModel | None = None


class AsrResponse(BaseModel):
    text: str
    mode: Mode

def _get_direct_model() -> tuple[FunASRNano, Dict[str, Any]]:
    global _direct_model_cache
    if _direct_model_cache is None:
        _direct_model_cache = cast(
            tuple[FunASRNano, Dict[str, Any]],
            FunASRNano.from_pretrained(model=DIRECT_MODEL_DIR, device="cuda:0"),
        )
        _direct_model_cache[0].eval()
    return _direct_model_cache



def _get_vad_model() -> AutoModel:
    global _vad_model_cache
    if _vad_model_cache is None:
        _vad_model_cache = AutoModel(
            model=ASR_VAD_MODEL_DIR,
            vad_model=VAD_MODEL_DIR,
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
            disable_update=True,
        )
    return _vad_model_cache


def _transcribe_direct(wav_path: str) -> str:
    model, kwargs = _get_direct_model()
    res = model.inference(data_in=[wav_path], **kwargs)
    return res[0][0]["text"]


def _transcribe_vad(wav_path: str) -> str:
    model = _get_vad_model()
    res = model.generate(input=[wav_path], cache={}, batch_size=1)
    return res[0]["text"]


def _transcribe(wav_path: str, mode: Mode) -> AsrResponse:
    if mode == MODE_VAD:
        text = _transcribe_vad(wav_path)
    else:
        text = _transcribe_direct(wav_path)
    return AsrResponse(text=text, mode=mode)


@app.get("/funasr/transcribe/path", response_model=AsrResponse)
def asr_path(
    wav_path: str,
    mode: Mode = Query(MODE_DIRECT),
) -> AsrResponse:
    if not Path(wav_path).is_file():
        raise HTTPException(status_code=404, detail="wav_path not found")
    return _transcribe(wav_path, mode)


@app.post("/funasr/transcribe", response_model=AsrResponse)
async def asr_upload(
    file: UploadFile = File(...),
    mode: Mode = Query(MODE_DIRECT),
) -> AsrResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        return _transcribe(tmp_path, mode)
    finally:
        Path(tmp_path).unlink(missing_ok=True)