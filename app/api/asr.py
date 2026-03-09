from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from app.api.schemas import AsrResponse
from app.domain.file_item import FileItem
from app.service.asr_service import get_stream_and_offline_models, transcribe_from_file_item, transcribe_from_path

api_router = APIRouter()

# Hardcoded streaming config (text3 style).
STREAM_CHUNK_SIZE = [0, 10, 5]
STREAM_ENCODER_CHUNK_LOOK_BACK = 4
STREAM_DECODER_CHUNK_LOOK_BACK = 1
STREAM_SAMPLE_RATE = 16000
STREAM_CHUNK_STRIDE = STREAM_CHUNK_SIZE[1] * 960


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
async def asr_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    stream_model, offline_model = get_stream_and_offline_models()

    cache: dict[str, object] = {}
    audio_buffer = bytearray()
    full_audio_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            chunk = message.get("bytes")
            if chunk:
                audio_buffer.extend(chunk)
                full_audio_buffer.extend(chunk)

                while len(audio_buffer) >= STREAM_CHUNK_STRIDE * 2:
                    pcm = audio_buffer[: STREAM_CHUNK_STRIDE * 2]
                    del audio_buffer[: STREAM_CHUNK_STRIDE * 2]

                    speech_chunk = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768

                    res = stream_model.generate(
                        input=speech_chunk,
                        cache=cache,
                        is_final=False,
                        chunk_size=STREAM_CHUNK_SIZE,
                        encoder_chunk_look_back=STREAM_ENCODER_CHUNK_LOOK_BACK,
                        decoder_chunk_look_back=STREAM_DECODER_CHUNK_LOOK_BACK,
                    )

                    if res and res[0].get("text"):
                        await websocket.send_json(
                            {
                                "type": "partial",
                                "text": res[0]["text"],
                            }
                        )
                continue

            text = message.get("text")
            if text != "end":
                continue

            if audio_buffer:
                speech_chunk = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768
                res = stream_model.generate(
                    input=speech_chunk,
                    cache=cache,
                    is_final=True,
                    chunk_size=STREAM_CHUNK_SIZE,
                    encoder_chunk_look_back=STREAM_ENCODER_CHUNK_LOOK_BACK,
                    decoder_chunk_look_back=STREAM_DECODER_CHUNK_LOOK_BACK,
                )
            else:
                res = stream_model.generate(
                    input=None,
                    cache=cache,
                    is_final=True,
                    chunk_size=STREAM_CHUNK_SIZE,
                    encoder_chunk_look_back=STREAM_ENCODER_CHUNK_LOOK_BACK,
                    decoder_chunk_look_back=STREAM_DECODER_CHUNK_LOOK_BACK,
                )

            await websocket.send_json(
                {
                    "type": "final",
                    "text": res[0]["text"] if res else "",
                }
            )

            if full_audio_buffer:
                full_audio = np.frombuffer(full_audio_buffer, dtype=np.int16).astype(np.float32) / 32768
                full_res = offline_model.generate(input=full_audio)
            else:
                full_res = offline_model.generate(input=np.array([], dtype=np.float32))

            await websocket.send_json(
                {
                    "type": "full",
                    "text": full_res[0]["text"] if full_res else "",
                }
            )
            return
    except WebSocketDisconnect:
        return


