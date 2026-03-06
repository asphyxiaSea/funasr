from pathlib import Path
import tempfile
from typing import Literal
import wave

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.api.schemas import AsrResponse, AsrStreamEvent, AsrWsControl
from app.config.settings import get_settings
from app.domain.file_item import FileItem
from app.service.asr_service import create_pcm16_stream_session, transcribe_from_file_item, transcribe_from_path

api_router = APIRouter()


def _pcm_to_wav(pcm_path: Path, wav_path: Path, sample_rate: int, channels: int) -> None:
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        with pcm_path.open("rb") as pcm_file:
            while True:
                chunk = pcm_file.read(4096)
                if not chunk:
                    break
                wf.writeframes(chunk)


EventType = Literal["partial", "final", "full", "error"]


async def _send_ws_event(websocket: WebSocket, event_type: EventType, text: str, is_final: bool) -> None:
    await websocket.send_json(
        AsrStreamEvent(type=event_type, text=text, is_final=is_final).model_dump(),
    )


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


@api_router.websocket("/funasr/transcribe/stream")
async def asr_stream_pcm16(
    websocket: WebSocket,
) -> None:
    settings = get_settings()
    expected_channels = settings.streaming_channels
    expected_sample_rate = settings.streaming_sample_rate

    await websocket.accept()

    session = create_pcm16_stream_session(sample_rate=expected_sample_rate)
    pcm_tmp = tempfile.NamedTemporaryFile(suffix=".pcm", delete=False)
    pcm_path = Path(pcm_tmp.name)
    pcm_tmp.close()
    wav_path: Path | None = None

    received_any = False
    pending_bytes = b""
    try:
        with pcm_path.open("wb") as pcm_file:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                chunk = message.get("bytes")
                if chunk is not None:
                    if not chunk:
                        continue
                    received_any = True

                    # Keep persisted stream bytes aligned to PCM16 frames.
                    data = pending_bytes + chunk
                    aligned_size = (len(data) // 2) * 2
                    aligned_chunk = data[:aligned_size]
                    pending_bytes = data[aligned_size:]

                    if aligned_chunk:
                        pcm_file.write(aligned_chunk)

                    text = session.feed(aligned_chunk)
                    if text:
                        await _send_ws_event(websocket, "partial", text, False)
                    continue

                text_message = message.get("text")
                if text_message is None:
                    await _send_ws_event(websocket, "error", "unsupported frame", True)
                    continue

                try:
                    AsrWsControl.model_validate_json(text_message)
                except ValidationError:
                    await _send_ws_event(websocket, "error", "unknown control message", False)
                    continue

                if not received_any:
                    await _send_ws_event(websocket, "error", "empty stream", True)
                    return

                final_text = session.finalize()
                await _send_ws_event(websocket, "final", final_text, True)

                wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                wav_path = Path(wav_tmp.name)
                wav_tmp.close()

                _pcm_to_wav(
                    pcm_path=pcm_path,
                    wav_path=wav_path,
                    sample_rate=expected_sample_rate,
                    channels=expected_channels,
                )
                full_response = transcribe_from_path(str(wav_path))
                await _send_ws_event(websocket, "full", full_response.text, True)
                return
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await _send_ws_event(websocket, "error", str(exc), True)
        except Exception:
            pass
    finally:
        pcm_path.unlink(missing_ok=True)
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
