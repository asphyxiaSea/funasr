from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel


MODE_ONLY_ASR: Literal["only_asr"] = "only_asr"
MODE_FUNASR: Literal["funasr"] = "funasr"
Mode = Literal["only_asr", "funasr"]


@dataclass(frozen=True)
class FileItem:
    filename: str
    content_type: str
    data: bytes


class AsrResponse(BaseModel):
    text: str
    mode: Mode

