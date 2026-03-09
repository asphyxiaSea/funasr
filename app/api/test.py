
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from funasr import AutoModel
import time

router = APIRouter(tags=["test"])

# streaming参数
chunk_size = [0, 10, 5]  # 600ms
encoder_chunk_look_back = 4
decoder_chunk_look_back = 1
sample_rate = 16000

chunk_stride = chunk_size[1] * 960  # 600ms

# 模型加载
asr_stream_model = AutoModel(model="models/ASRmodels/paraformer-zh-streaming",
                             disable_update=True)

# 全量模型（通常用非streaming模型）
funasr_model = AutoModel(model="models/ASRmodels/paraformer-zh",
                      vad_model="models/VADmodels/speech_fsmn_vad_zh",
                      punc_model="models/PUNCmodels/ct-punc",
                      spk_model="models/SPKmodels/cam++",
                      disable_update=True)



@router.websocket("/funasr/transcribe/stream/test")
async def asr_stream(ws: WebSocket):
    await ws.accept()

    cache = {}
    audio_buffer = bytearray()
    full_audio_buffer = bytearray()  # 保存完整音频

    try:
        while True:
            data = await ws.receive()

            # binary audio
            chunk = data.get("bytes")
            if chunk:
                audio_buffer.extend(chunk)
                full_audio_buffer.extend(chunk)

                while len(audio_buffer) >= chunk_stride * 2:
                    pcm = audio_buffer[:chunk_stride * 2]
                    del audio_buffer[:chunk_stride * 2]

                    speech_chunk = (
                        np.frombuffer(pcm, dtype=np.int16)
                        .astype(np.float32) / 32768
                    )
                    t1 = time.perf_counter()
                    res = asr_stream_model.generate(
                        input=speech_chunk,
                        cache=cache,
                        is_final=False,
                        chunk_size=chunk_size,
                        encoder_chunk_look_back=encoder_chunk_look_back,
                        decoder_chunk_look_back=decoder_chunk_look_back,
                    )
                    t2 = time.perf_counter()
                    print("stream infer0:", t2 - t1)
                    if res and res[0]["text"]:
                        await ws.send_json({
                            "type": "partial",
                            "text": res[0]["text"],
                        })

                continue

            # control message
            text = data.get("text")
            if text == "end":

                # flush streaming
                if audio_buffer:
                    speech_chunk = (
                        np.frombuffer(audio_buffer, dtype=np.int16)
                        .astype(np.float32) / 32768
                    )

                    res = asr_stream_model.generate(
                        input=speech_chunk,
                        cache=cache,
                        is_final=True,
                        chunk_size=chunk_size,
                        encoder_chunk_look_back=encoder_chunk_look_back,
                        decoder_chunk_look_back=decoder_chunk_look_back,
                    )
                else:
                    res = asr_stream_model.generate(
                        input=None,
                        cache=cache,
                        is_final=True,
                        chunk_size=chunk_size,
                        encoder_chunk_look_back=encoder_chunk_look_back,
                        decoder_chunk_look_back=decoder_chunk_look_back,
                    )

                final_text = res[0]["text"] if res else ""

                await ws.send_json({
                    "type": "final",
                    "text": final_text
                })

                # ---------- 全量复跑 ----------
                full_audio = (
                    np.frombuffer(full_audio_buffer, dtype=np.int16)
                    .astype(np.float32) / 32768
                )

                full_res = funasr_model.generate(input=full_audio)

                await ws.send_json({
                    "type": "full",
                    "text": full_res[0]["text"] if full_res else ""
                })

                return

    except WebSocketDisconnect:
        return
