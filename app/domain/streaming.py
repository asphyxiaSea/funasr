from dataclasses import dataclass, field


@dataclass
class StreamSession:
    cache: dict = field(default_factory=dict)
    audio_buffer: bytearray = field(default_factory=bytearray)
    full_audio_buffer: bytearray = field(default_factory=bytearray)

    def append_chunk(self, chunk: bytes) -> None:
        self.audio_buffer.extend(chunk)
        self.full_audio_buffer.extend(chunk)

    def pop_stream_frame(self, frame_bytes: int) -> bytes | None:
        if len(self.audio_buffer) < frame_bytes:
            return None
        frame = bytes(self.audio_buffer[:frame_bytes])
        del self.audio_buffer[:frame_bytes]
        return frame
