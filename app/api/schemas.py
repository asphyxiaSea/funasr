from typing import Literal

from pydantic import BaseModel


class AsrResponse(BaseModel):
    text: str


class AsrStreamEvent(BaseModel):
    type: Literal["partial", "final", "full", "error"]
    text: str
    is_final: bool = False


class AsrWsControl(BaseModel):
    type: Literal["end"]


class SpeakerEmbeddingResponse(BaseModel):
    embedding_b64: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ready: bool
    detail: str | None = None
