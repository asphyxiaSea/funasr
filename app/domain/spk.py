from pydantic import BaseModel

from app.domain.file_item import FileItem


class SpeakerEmbeddingResponse(BaseModel):
    embedding_b64: str
