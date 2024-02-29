from fastapi import APIRouter
from fastapi import HTTPException

from ..models import EmbeddingRead, EmbeddingCreateMulti, EmbeddingPagination, EmbeddingDelete

from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams
from embeddings.app.deps.request_params import CommonParams

from .service import (
    insert,
    collection_exists,
    embeddings,
    delete,
    embedding
)

from embeddings.models import InsertionResult

from embeddings.app.config import settings


router = APIRouter(prefix="/embeddings/qdrant")


@router.get("/{namespace}/{embedding_id}", response_model=EmbeddingRead)
async def get_embedding(namespace: str, embedding_id: str, ):
    client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT)
    return await embedding(
        client=client,
        namespace=namespace,
        embedding_id=embedding_id
    )


@router.get("/{namespace}", response_model=EmbeddingPagination)
async def get_embeddings(namespace: str, common: CommonParams, ):
    client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT)
    return await embeddings(
        client=client,
        namespace=namespace,
        common=common
    )


@router.post("/{namespace}", response_model=InsertionResult[EmbeddingRead])
async def create_embedding(namespace: str, data_in: EmbeddingCreateMulti, ):
    client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT)
    exists = await collection_exists(client, namespace)
    if not exists and not data_in.create_namespace:
        raise HTTPException(
            status_code=404,
            detail=[{"msg": f"Collection with name {namespace} does not exist"}]
        )
    elif not exists:
        vector_size = data_in.embedding_model.dimensionality
        await client.create_collection(
            collection_name=namespace,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ),
        )

    res = await insert(
        client=client,
        data_in=data_in,
        namespace=namespace,
    )
    return res


@router.delete("/{namespace}/{embedding_id}", response_model=EmbeddingDelete)
async def delete_embedding(namespace: str, embedding_id: str, ):
    client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT)
    return await delete(
        client=client,
        namespace=namespace,
        embedding_ids=[embedding_id]
    )
