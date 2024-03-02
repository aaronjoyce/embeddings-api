from typing import Annotated

from fastapi import Depends

from qdrant_client.async_qdrant_client import AsyncQdrantClient

from app.config import settings


def qdrant_api_client():
    yield AsyncQdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_HTTP_PORT
    )


QdrantClient = Annotated[AsyncQdrantClient, Depends(qdrant_api_client)]
