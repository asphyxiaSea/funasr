from fastapi import APIRouter

from .health import router as health_router
from .speaker import router as speaker_router
from .transcribe import router as transcribe_router
from .test import router as test_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(transcribe_router)
api_router.include_router(speaker_router)
api_router.include_router(test_router)