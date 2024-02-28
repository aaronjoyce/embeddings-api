from fastapi import status
from fastapi import HTTPException

from embeddings.app.lib.cloudflare.api import API, CloudflareEmbeddingModels

from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from qdrant_client.http.models import VectorParams

from embeddings.app.config import settings
from embeddings.app.deps.request_params import CommonParams

from embeddings.app.document.models import DocumentRead, DocumentPagination

from .models import NamespaceRead
from .models import NamespaceCreate
from .models import NamespaceDelete
from ..models import NamespaceQuery


async def namespace(name: str) -> NamespaceRead:
    client = AsyncQdrantClient(host=settings.QDRANT_HOST)
    try:
        result = await client.get_collection(
            collection_name=name
        )
    except UnexpectedResponse as ex:
        if ex.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=[{"msg": f"Collection with name {namespace} not found"}]
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=[{"msg": ex.content.decode('utf-8')}]
        )

    data = {
        "name": name,
        "dimensionality": result.config.params.vectors.size,
        "distance": str(result.config.params.vectors.distance),
        "status": str(result.status),
        "shard_number": result.config.params.shard_number,
        "replication_factor": result.config.params.replication_factor,
        "write_consistency_factor": result.config.params.write_consistency_factor,
        "vectors_count": result.vectors_count,
        "points_count": result.points_count
    }
    return NamespaceRead(
        **data
    )


async def create(data_in: NamespaceCreate) -> NamespaceRead:
    client = AsyncQdrantClient(host=settings.QDRANT_HOST)
    result = await client.create_collection(
        collection_name=data_in.name,
        vectors_config=VectorParams(size=data_in.dimensionality, distance=data_in.distance),
    )
    return await namespace(name=data_in.name)


async def delete(name: str) -> NamespaceDelete:
    client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT)
    try:
        deletion_res = await client.delete_collection(
            collection_name=name
        )
    except UnexpectedResponse as ex:
        if ex.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=[{"msg": f"The namespace {namespace} you provided does not exist"}]
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=[{"msg": ex.content.decode('utf-8')}]
        )

    return NamespaceDelete(
        success=True
    )


async def embedding_matches(namespace: str, data_in: NamespaceQuery, common: CommonParams):
    client = API(
        api_token=settings.CLOUDFLARE_API_TOKEN,
        account_id=settings.CLOUDFLARE_API_ACCOUNT_ID
    )
    res = client.embed(
        model=CloudflareEmbeddingModels.BAAIBase.value,
        texts=[data_in.inputs]
    )
    query_vectors = res.get('data', [])
    query_vector = query_vectors[0]

    client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT)
    query_search_result = await client.search(
        collection_name=namespace,
        query_vector=query_vector,
        offset=common.get("offset"),
        limit=common.get("limit")
    )
    data = {
        "items": [DocumentRead(id=o.id, payload=o.payload, score=o.score, vector=o.vector) for o in
                  query_search_result],
        "total": len(query_search_result),
        "page": common.get("page"),
    }
    return DocumentPagination(**data)
