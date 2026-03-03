from typing import Literal

from pydantic import BaseModel


MODE_ONLY_ASR: Literal["only_asr"] = "only_asr"
MODE_FUNASR: Literal["funasr"] = "funasr"
Mode = Literal["only_asr", "funasr"]


class AsrResponse(BaseModel):
    text: str
    mode: Mode


class SpeakerEmbeddingResponse(BaseModel):
    embedding_b64: str
