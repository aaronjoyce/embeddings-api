from fastapi import APIRouter, Depends

from .embeddings.views import router

api_router = APIRouter(
    prefix="/api/v1"
)

api_router.include_router(
    router
)
