from fastapi import APIRouter
from fastapi import Request
from fastapi import Response
from fastapi import HTTPException

from ..models import EmbeddingRead, EmbeddingCreateMulti, EmbeddingPagination

from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams
from embeddings.app.deps.request_params import CommonParams

from .service import embedding as get_embedding
from .service import (
    insert_embedding,
    collection_exists,
    embeddings,
)

from embeddings.models import InsertionResult

from embeddings.app.config import settings


router = APIRouter(prefix="/embeddings/qdrant")

client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT)


@router.get("/{namespace}/{embedding_id}", response_model=EmbeddingRead)
async def get(namespace: str, embedding_id: str, ):
    return await get_embedding(
        client=client,
        namespace=namespace,
        embedding_id=embedding_id
    )


@router.get("/{namespace}", response_model=EmbeddingPagination)
async def list_embeddings(namespace: str, common: CommonParams, ):
    return await embeddings(
        client=client,
        namespace=namespace,
        common=common
    )


@router.post("/{namespace}", response_model=InsertionResult[EmbeddingRead])
async def create(namespace: str, data_in: EmbeddingCreateMulti, ):
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

    res = await insert_embedding(
        client=client,
        data_in=data_in,
        namespace=namespace,
    )
    return res
