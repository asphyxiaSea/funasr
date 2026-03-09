import asyncio
import wave
import websockets

WS_URL = "ws://127.0.0.1:8010/funasr/transcribe/stream"
WAV_FILE = "assets/asr_example.wav"

SAMPLE_RATE = 16000
CHUNK_MS = 600


async def send_audio():
    async with websockets.connect(WS_URL, max_size=None) as ws:

        async def receiver():
            try:
                async for msg in ws:
                    print("SERVER:", msg)
            except websockets.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(receiver())

        with wave.open(WAV_FILE, "rb") as wf:

            if wf.getframerate() != SAMPLE_RATE:
                raise ValueError("wav must be 16k sample rate")

            if wf.getnchannels() != 1:
                raise ValueError("wav must be mono")

            if wf.getsampwidth() != 2:
                raise ValueError("wav must be PCM16")

            frames_per_chunk = int(SAMPLE_RATE * CHUNK_MS / 1000)

            while True:
                frames = wf.readframes(frames_per_chunk)

                if not frames:
                    break

                await ws.send(frames)

                await asyncio.sleep(CHUNK_MS / 1000)

        print("audio finished")

        await ws.send("end")

        await recv_task


if __name__ == "__main__":
    asyncio.run(send_audio())