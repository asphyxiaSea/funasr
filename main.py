import logging

import uvicorn
from fastapi import FastAPI

from app.api import api_router
from app.config.settings import get_settings
from app.domain.models import initialize_models

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="FunASR Service")
    app.include_router(api_router)

    @app.on_event("startup")
    def _startup() -> None:
        settings = get_settings()
        try:
            initialize_models(settings)
        except Exception:
            logger.exception("Failed to initialize models on startup")

    return app


app = create_app()


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8010, workers=1)


if __name__ == "__main__":
    main()
