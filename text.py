import asyncio
import wave
import json
import websockets

WS_URL = "ws://127.0.0.1:8010/funasr/transcribe/stream"
WAV_FILE = "assets/speaker1_b_cn_16k.wav"

CHUNK_MS = 500  # 每次发送500ms音频
FINAL_TIMEOUT_SEC = 120


async def stream_wav():
    with wave.open(WAV_FILE, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()

        if sampwidth != 2:
            raise ValueError("Only PCM16 wav supported")

        print("sample_rate:", sample_rate)
        print("channels:", channels)

        # 500ms chunk
        frames_per_chunk = int(sample_rate * CHUNK_MS / 1000)

        async with websockets.connect(WS_URL) as ws:
            print("connected")

            terminal_event = asyncio.Event()
            terminal_payload: dict[str, str | None] = {"type": None, "text": None}

            async def receiver():
                try:
                    async for message in ws:
                        print("SERVER:", message)
                        try:
                            payload = json.loads(message)
                        except json.JSONDecodeError:
                            continue

                        event_type = payload.get("type")
                        if event_type in ("full", "error"):
                            terminal_payload["type"] = event_type
                            text = payload.get("text")
                            terminal_payload["text"] = text if isinstance(text, str) else None
                            terminal_event.set()
                            return
                except websockets.ConnectionClosed:
                    print(f"SERVER CLOSED: code={ws.close_code}, reason={ws.close_reason}")
                    terminal_payload["type"] = "closed"
                    terminal_event.set()

            recv_task = asyncio.create_task(receiver())

            while True:
                frames = wf.readframes(frames_per_chunk)
                if not frames:
                    break

                # 发送PCM bytes
                await ws.send(frames)

                # 模拟实时
                await asyncio.sleep(CHUNK_MS / 1000)

            print("audio finished")

            # 发送结束控制消息
            await ws.send(json.dumps({"type": "end"}))

            try:
                await asyncio.wait_for(terminal_event.wait(), timeout=FINAL_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                print(f"Timeout: no full/error event within {FINAL_TIMEOUT_SEC}s")

            if not recv_task.done():
                recv_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await recv_task

            print("terminal:", terminal_payload)


if __name__ == "__main__":
    import contextlib

    asyncio.run(stream_wav())