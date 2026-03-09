from fastapi import APIRouter

from app.domain.models import get_init_error, is_ready

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    ready = is_ready()
    return {
        "ok": ready,
        "error": None if ready else (get_init_error() or "models not initialized"),
    }
