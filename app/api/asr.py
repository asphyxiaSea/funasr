from pathlib import Path
import json

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.api.schemas import AsrResponse, AsrStreamEvent
from app.config.settings import get_settings
from app.domain.file_item import FileItem
from app.service.asr_service import create_pcm16_stream_session, transcribe_from_file_item, transcribe_from_path

api_router = APIRouter()


def _format_sse(event: AsrStreamEvent) -> bytes:
    payload = json.dumps(event.model_dump(), ensure_ascii=False)
    return f"event: {event.type}\ndata: {payload}\n\n".encode("utf-8")


@api_router.get("/funasr/transcribe/path", response_model=AsrResponse)
def asr_path(
    wav_path: str,
) -> AsrResponse:
    if not Path(wav_path).is_file():
        raise HTTPException(status_code=404, detail="wav_path not found")
    return transcribe_from_path(wav_path)


@api_router.post("/funasr/transcribe", response_model=AsrResponse)
async def asr_upload(
    file: UploadFile = File(...),
) -> AsrResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    file_item = FileItem(
        filename=file.filename or "audio.wav",
        content_type=file.content_type or "application/octet-stream",
        data=data,
    )
    return transcribe_from_file_item(file_item)


@api_router.post("/funasr/transcribe/stream")
async def asr_stream_pcm16(
    request: Request,
    sample_rate: int | None = Query(None),
    channels: int | None = Query(None),
) -> StreamingResponse:
    settings = get_settings()
    expected_channels = settings.streaming_channels
    expected_sample_rate = sample_rate or settings.streaming_sample_rate

    if channels is not None and channels != expected_channels:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported channels={channels}, expected {expected_channels}",
        )
    if expected_sample_rate != settings.streaming_sample_rate:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported sample_rate={expected_sample_rate}, "
                f"expected {settings.streaming_sample_rate}"
            ),
        )

    async def event_generator():
        session = create_pcm16_stream_session(sample_rate=expected_sample_rate)
        received_any = False
        try:
            async for chunk in request.stream():
                if not chunk:
                    continue
                received_any = True
                text = session.feed(chunk)
                if text:
                    yield _format_sse(
                        AsrStreamEvent(type="partial", text=text, is_final=False),
                    )

            if not received_any:
                yield _format_sse(
                    AsrStreamEvent(type="error", text="empty stream", is_final=True),
                )
                return

            final_text = session.finalize()
            yield _format_sse(
                AsrStreamEvent(type="final", text=final_text, is_final=True),
            )
        except Exception as exc:
            yield _format_sse(
                AsrStreamEvent(type="error", text=str(exc), is_final=True),
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
