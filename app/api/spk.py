from fastapi import APIRouter, File, HTTPException, UploadFile

from app.domain.file_item import FileItem
from app.domain.spk import SpeakerEmbeddingResponse
from app.service.spk_service import infer_from_file_item

api_router = APIRouter()


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
