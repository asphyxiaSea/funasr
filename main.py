from fastapi import FastAPI

from app.api import api_router
from app.config.settings import get_settings
from app.domain.loaders.funasr_loader import load_models
from app.service.asr_service import init_models


def create_app() -> FastAPI:
	app = FastAPI(title="FunASR Service")
	app.include_router(api_router)

	@app.on_event("startup")
	def _startup() -> None:
		settings = get_settings()
		bundle = load_models(settings)
		init_models(bundle)

	return app


app = create_app()