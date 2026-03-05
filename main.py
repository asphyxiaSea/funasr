import logging

from fastapi import FastAPI

from app.api import api_router
from app.config.settings import get_settings
from app.domain.funasr_loader import load_models
from app.service.asr_service import init_models, set_init_error


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
	app = FastAPI(title="FunASR Service")
	app.include_router(api_router)

	@app.on_event("startup")
	def _startup() -> None:
		try:
			settings = get_settings()
			bundle = load_models(settings)
			init_models(bundle)
		except Exception as exc:
			logger.exception("Failed to initialize models on startup")
			set_init_error(str(exc))

	return app


app = create_app()