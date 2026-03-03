from fastapi import APIRouter

from app.api.asr import api_router as asr_router
from app.api.spk import api_router as spk_router

api_router = APIRouter()
api_router.include_router(asr_router)
api_router.include_router(spk_router)

__all__ = ["api_router"]
