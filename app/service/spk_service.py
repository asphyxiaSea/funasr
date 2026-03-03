import base64
from functools import lru_cache
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from funasr import AutoModel

from app.api.schemas import SpeakerEmbeddingResponse
from app.config.settings import get_settings
from app.domain.file_item import FileItem


def _embedding_to_base64(embedding: torch.Tensor) -> str:
    normalized = F.normalize(embedding, p=2, dim=1)
    vector = normalized.detach().cpu().numpy().astype(np.float32)
    return base64.b64encode(vector.tobytes()).decode("utf-8")


@lru_cache(maxsize=1)
def _get_model() -> Any:
    settings = get_settings()
    return AutoModel(
        model=settings.funasr_spk_model_dir,
        device=settings.device,
        disable_update=True,
    )


def _infer_path(wav_path: str) -> SpeakerEmbeddingResponse:
    model = _get_model()
    result = model.generate(input=wav_path)
    embedding = result[0]["spk_embedding"]
    b64 = _embedding_to_base64(embedding)
    return SpeakerEmbeddingResponse(embedding_b64=b64)


def infer_from_path(wav_path: str) -> SpeakerEmbeddingResponse:
    return _infer_path(wav_path)


def infer_from_file_item(file_item: FileItem) -> SpeakerEmbeddingResponse:
    suffix = Path(file_item.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_item.data)
        tmp_path = tmp.name

    try:
        return _infer_path(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
