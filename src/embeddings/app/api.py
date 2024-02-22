from fastapi import APIRouter, Depends

from .embeddings.views import router as embeddings_router
from .namespace.views import router as namespace_router

api_router = APIRouter(
    prefix="/api/v1"
)

api_router.include_router(
    embeddings_router
)
api_router.include_router(
    namespace_router
)
