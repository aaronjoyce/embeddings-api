from fastapi import APIRouter, Depends

from .embeddings.cloudflare.views import router as cloudflare_embeddings_router
from .embeddings.qdrant.views import router as qdrant_embeddings_router
from .namespace.qdrant.views import router as qdrant_namespace_router
from .namespace.cloudflare.views import router as cloudflare_namespace_router


api_router = APIRouter(
    prefix="/api/v1"
)

api_router.include_router(
    cloudflare_embeddings_router
)
api_router.include_router(
    qdrant_embeddings_router
)
api_router.include_router(
    qdrant_namespace_router
)
api_router.include_router(
    cloudflare_namespace_router
)
