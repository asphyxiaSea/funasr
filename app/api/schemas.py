from typing import Literal

from pydantic import BaseModel


class AsrResponse(BaseModel):
    text: str


class SpeakerEmbeddingResponse(BaseModel):
    embedding_b64: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ready: bool
    detail: str | None = None
