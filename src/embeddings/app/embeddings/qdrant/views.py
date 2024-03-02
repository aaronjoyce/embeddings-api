from fastapi import APIRouter, status

from ..models import EmbeddingRead, EmbeddingCreateMulti, EmbeddingPagination, EmbeddingDelete

from embeddings.app.deps.request_params import CommonParams
from embeddings.app.deps.qdrant import QdrantClient

from .service import (
    embeddings,
    delete,
    embedding,
    create
)

from embeddings.models import InsertionResult


router = APIRouter(prefix="/embeddings/qdrant")


@router.get("/{namespace}/{embedding_id}", response_model=EmbeddingRead)
async def get_embedding(namespace: str, embedding_id: str, client: QdrantClient):
    return await embedding(
        client=client,
        namespace=namespace,
        embedding_id=embedding_id
    )


@router.get("/{namespace}", response_model=EmbeddingPagination)
async def get_embeddings(namespace: str, common: CommonParams, client: QdrantClient):
    return await embeddings(
        client=client,
        namespace=namespace,
        common=common
    )


@router.post("/{namespace}", response_model=InsertionResult[EmbeddingRead], status_code=status.HTTP_201_CREATED)
async def create_embedding(namespace: str, data_in: EmbeddingCreateMulti, client: QdrantClient):
    return await create(client=client, namespace=namespace, data_in=data_in)


@router.delete("/{namespace}/{embedding_id}", response_model=EmbeddingDelete)
async def delete_embedding(namespace: str, embedding_id: str, client: QdrantClient):
    return await delete(
        client=client,
        namespace=namespace,
        embedding_ids=[embedding_id]
    )
