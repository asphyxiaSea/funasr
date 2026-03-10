import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from funasr import AutoModel
import uvicorn
import time
import tempfile
import shutil 
import numpy as np
import torch
import torch.nn.functional as F
import base64


app = FastAPI()

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

# 说话人验证模型（如果需要单独调用说话人验证功能，可以在这里加载）
spk_embedding_model = AutoModel(model="models/SPKmodels/cam++", disable_update=True)



@app.get("/funasr/transcribe/path")
def asr_path(
    wav_path: str,
) -> str:
    res = funasr_model.generate(input=wav_path)
    return res[0]["text"] if res else ""



def _embedding_to_base64(embedding: torch.Tensor) -> str:
    normalized = F.normalize(embedding, p=2, dim=1)
    vector = normalized.detach().cpu().numpy().astype(np.float32)
    return base64.b64encode(vector.tobytes()).decode("utf-8")


@app.get("/funasr/spk/embedding")
def spk_embedding(
    file: UploadFile = File(...),
) -> str:
    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    res = spk_embedding_model.generate(input=tmp_path)
    embedding = res[0]["spk_embedding"]
    norm_embedding = F.normalize(embedding, p=2, dim=1)
    vector  = norm_embedding.detach().cpu().numpy().astype(np.float32)
    embedding_base64 = base64.b64encode(vector.tobytes()).decode("utf-8")
    return embedding_base64


@app.post("/funasr/transcribe")
async def asr_upload(
file: UploadFile = File(...),
) -> str:
    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # FunASR 使用文件路径
    res = funasr_model.generate(input=tmp_path)

    return res[0]["text"] if res else ""


@app.websocket("/funasr/transcribe/stream")
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
                    "results": full_res if full_res else ""
                })

                return

    except WebSocketDisconnect:
        return


def main():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8010,
        workers=1
    )


if __name__ == "__main__":
    main()