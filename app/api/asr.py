from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.domain.asr import AsrResponse, FileItem, Mode, MODE_DIRECT
from app.service.asr_service import transcribe_from_file_item, transcribe_from_path

api_router = APIRouter()


@api_router.get("/funasr/transcribe/path", response_model=AsrResponse)
def asr_path(
    wav_path: str,
    mode: Mode = Query(MODE_DIRECT),
) -> AsrResponse:
    if not Path(wav_path).is_file():
        raise HTTPException(status_code=404, detail="wav_path not found")
    return transcribe_from_path(wav_path, mode)


@api_router.post("/funasr/transcribe", response_model=AsrResponse)
async def asr_upload(
    file: UploadFile = File(...),
    mode: Mode = Query(MODE_DIRECT),
) -> AsrResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    file_item = FileItem(
        filename=file.filename or "audio.wav",
        content_type=file.content_type or "application/octet-stream",
        data=data,
    )
    return transcribe_from_file_item(file_item, mode)
