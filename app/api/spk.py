from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import SpeakerEmbeddingResponse
from app.domain.file_item import FileItem
from app.service.spk_service import infer_from_file_item, infer_from_path

api_router = APIRouter()


@api_router.get("/funasr/spk/embedding/path", response_model=SpeakerEmbeddingResponse)
def create_speaker_embedding_path(
    wav_path: str,
) -> SpeakerEmbeddingResponse:
    if not Path(wav_path).is_file():
        raise HTTPException(status_code=404, detail="wav_path not found")
    return infer_from_path(wav_path)


@api_router.post("/funasr/spk/embedding", response_model=SpeakerEmbeddingResponse)
async def create_speaker_embedding(
    file: UploadFile = File(...),
) -> SpeakerEmbeddingResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    file_item = FileItem(
        filename=file.filename or "audio.wav",
        content_type=file.content_type or "application/octet-stream",
        data=data,
    )
    return infer_from_file_item(file_item)
