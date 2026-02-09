from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel


MODE_DIRECT: Literal["direct"] = "direct"
MODE_VAD: Literal["vad"] = "vad"
Mode = Literal["direct", "vad"]


@dataclass(frozen=True)
class FileItem:
    filename: str
    content_type: str
    data: bytes


class AsrResponse(BaseModel):
    text: str
    mode: Mode

