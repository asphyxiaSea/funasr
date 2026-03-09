from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from app.api.schemas import TranscribeResponse
from app.config.settings import get_settings
from app.domain.streaming import StreamSession
from app.service.transcribe_service import (
    finalize_stream,
    process_stream_bytes,
    rerun_full_audio,
    transcribe_path,
    transcribe_upload,
)

router = APIRouter(tags=["transcribe"])


@router.get("/funasr/transcribe/path", response_model=TranscribeResponse)
def asr_path(wav_path: str) -> TranscribeResponse:
    if not Path(wav_path).is_file():
        raise HTTPException(status_code=404, detail="wav_path not found")
    return TranscribeResponse(text=transcribe_path(wav_path))


@router.post("/funasr/transcribe", response_model=TranscribeResponse)
async def asr_upload(file: UploadFile = File(...)) -> TranscribeResponse:
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    return TranscribeResponse(
        text=transcribe_upload(file_bytes=file_bytes, filename=file.filename or "audio.wav")
    )


@router.websocket("/funasr/transcribe/stream")
async def asr_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    settings = get_settings()
    session = StreamSession()

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                return

            chunk = message.get("bytes")
            if chunk:
                partials = process_stream_bytes(session=session, chunk_bytes=chunk, settings=settings)
                for text in partials:
                    await websocket.send_json({"type": "partial", "text": text})
                continue

            if message.get("text") != "end":
                continue

            final_text = finalize_stream(session=session, settings=settings)
            await websocket.send_json({"type": "final", "text": final_text})

            full_result = rerun_full_audio(session=session)
            await websocket.send_json({"type": "full", "result": full_result})
            return

    except WebSocketDisconnect:
        return
