from typing import Literal
from pydantic import BaseModel

from app.domain.file_item import FileItem


MODE_ONLY_ASR: Literal["only_asr"] = "only_asr"
MODE_FUNASR: Literal["funasr"] = "funasr"
Mode = Literal["only_asr", "funasr"]


class AsrResponse(BaseModel):
    text: str
    mode: Mode

