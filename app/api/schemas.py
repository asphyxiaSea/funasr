from pydantic import BaseModel


class TranscribeResponse(BaseModel):
    text: str


class StreamMessage(BaseModel):
    type: str
    text: str


class SpeakerEmbeddingResponse(BaseModel):
    embedding_b64: str
