from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import SpeakerEmbeddingResponse
from app.service.asr_service import spk_embedding_upload

router = APIRouter(tags=["speaker"])


@router.post("/funasr/speaker/embedding", response_model=SpeakerEmbeddingResponse)
async def spk_embedding(file: UploadFile = File(...)) -> SpeakerEmbeddingResponse:
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    return SpeakerEmbeddingResponse(
        embedding_b64=spk_embedding_upload(file_bytes=file_bytes, filename=file.filename or "audio.wav")
    )
