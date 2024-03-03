from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import settings

from .api import api_router


def create_app():
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        openapi_url=f"{settings.API_PATH}/openapi.json",
    )
    app.include_router(api_router)
    return app
