from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, cast

from funasr import AutoModel

from app.config.settings import Settings


@dataclass(frozen=True)
class ModelBundle:
    direct_model: Any
    direct_kwargs: Dict[str, Any]
    vad_model: AutoModel


def _ensure_funasr_path(funasr_dir: Path) -> None:
    path_str = str(funasr_dir)
    if path_str not in sys.path:
        sys.path.append(path_str)


def load_models(settings: Settings) -> ModelBundle:
    _ensure_funasr_path(settings.funasr_dir)

    from FunASR.model import FunASRNano

    direct_model, direct_kwargs = cast(
        tuple[FunASRNano, Dict[str, Any]],
        FunASRNano.from_pretrained(model=settings.direct_model_dir, device=settings.device),
    )
    direct_model.eval()

    vad_model = AutoModel(
        model=settings.asr_vad_model_dir,
        vad_model=settings.vad_model_dir,
        vad_kwargs={"max_single_segment_time": 30000},
        device=settings.device,
        disable_update=True,
    )

    return ModelBundle(
        direct_model=direct_model,
        direct_kwargs=direct_kwargs,
        vad_model=vad_model,
    )
