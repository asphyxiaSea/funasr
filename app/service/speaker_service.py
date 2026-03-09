import base64
import tempfile
from pathlib import Path

import numpy as np

from app.domain.models import get_models


def _embedding_to_base64(embedding: np.ndarray | list[float]) -> str:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        return ""

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return base64.b64encode(vector.tobytes()).decode("utf-8")


def _normalize_embedding_input(raw_embedding):
    # Support tensor-like outputs and plain arrays/lists.
    if hasattr(raw_embedding, "detach"):
        raw_embedding = raw_embedding.detach()
    if hasattr(raw_embedding, "cpu"):
        raw_embedding = raw_embedding.cpu()
    if hasattr(raw_embedding, "numpy"):
        raw_embedding = raw_embedding.numpy()
    return raw_embedding


def speaker_embedding_path(wav_path: str) -> str:
    models = get_models()
    result = models.spk_model.generate(input=wav_path)
    embedding = result[0].get("spk_embedding") if result else None
    if embedding is None:
        return ""
    return _embedding_to_base64(_normalize_embedding_input(embedding))


def speaker_embedding_upload(file_bytes: bytes, filename: str) -> str:
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        return speaker_embedding_path(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
