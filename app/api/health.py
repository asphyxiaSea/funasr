from fastapi import APIRouter

from app.api.schemas import HealthResponse
from app.service.asr_service import get_model_status

api_router = APIRouter()


@api_router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ready, detail = get_model_status()
    if ready:
        return HealthResponse(status="ok", ready=True, detail=None)
    return HealthResponse(status="degraded", ready=False, detail=detail)
