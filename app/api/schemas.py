from typing import Literal

from pydantic import BaseModel


MODE_FUNASR: Literal["funasr"] = "funasr"
Mode = Literal["funasr"]


class AsrResponse(BaseModel):
    text: str
    mode: Mode


class AsrStreamEvent(BaseModel):
    type: Literal["partial", "final", "error"]
    text: str
    mode: Mode = MODE_FUNASR
    is_final: bool = False


class SpeakerEmbeddingResponse(BaseModel):
    embedding_b64: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ready: bool
    detail: str | None = None
