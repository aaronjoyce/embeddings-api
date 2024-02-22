import traceback
from typing import Optional
from fastapi import APIRouter
from fastapi import Request
from fastapi import Response
from fastapi import HTTPException
from fastapi import status

from http import HTTPStatus

from .models import CreateNamespaceResponse
from .models import CreateNamespacePayload
from .models import RetrieveNamespaceData
from .models import DeleteNamespaceResponse
from .models import NamespaceQueryPayload

from embeddings.app.document.models import DocumentPagination
from embeddings.app.document.models import DocumentRead

from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from embeddings.app.config import settings

from embeddings.app.config import CloudflareEmbeddingModels

from embeddings.app.lib.cloudflare import embed

from embeddings.app.deps.request_params import CommonParams

router = APIRouter(prefix="/namespace")

client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT)


@router.post("", response_model=CreateNamespaceResponse)
async def create(data_in: CreateNamespacePayload, request: Request, response: Response):
    result = await client.create_collection(
        collection_name=data_in.name,
        vectors_config=VectorParams(size=data_in.dimensionality, distance=data_in.distance),
    )
    print(("result", result))
    return CreateNamespaceResponse(
        success=True
    )


@router.post("/{namespace}/query", response_model=DocumentPagination)
async def query(namespace: str, payload: NamespaceQueryPayload, common: CommonParams, request: Request, response: Response):
    embedded_query_result = embed(model=CloudflareEmbeddingModels.BAAIBase.value, text=[payload.inputs])
    query_vectors = embedded_query_result.get('result', {}).get('data', [])
    query_vector = query_vectors[0]
    query_search_result = await client.search(
        collection_name=namespace,
        query_vector=query_vector,
        offset=common.get("offset"),
        limit=common.get("limit")
    )
    print(("query_search_result", query_search_result))

    data = {
        "items": [DocumentRead(id=o.id, payload=o.payload, score=o.score, vector=o.vector) for o in query_search_result],
        "total": len(query_search_result),
        "page": 1
    }
    print(("query_search.data", data))
    return DocumentPagination(**data)


@router.get("/{namespace}", response_model=RetrieveNamespaceData)
async def get(namespace: str, request: Request, response: Response):
    try:
        result = await client.get_collection(
            collection_name=namespace
        )
    except UnexpectedResponse as ex:
        if ex.status_code == HTTPStatus.NOT_FOUND.value:
            response.status_code = HTTPStatus.NOT_FOUND.value
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=[{"msg": f"Collection with name {namespace} not found"}]
            )

        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=[{"msg": ex.content.decode('utf-8')}]
        )

    data = {
        "name": namespace,
        "dimensionality": result.config.params.vectors.size,
        "distance": str(result.config.params.vectors.distance),
        "status": str(result.status),
        "shard_number": result.config.params.shard_number,
        "replication_factor": result.config.params.replication_factor,
        "write_consistency_factor": result.config.params.write_consistency_factor,
        "vectors_count": result.vectors_count,
        "points_count": result.points_count
    }
    return RetrieveNamespaceData(
       **data
    )


@router.delete("/{namespace}", response_model=DeleteNamespaceResponse)
async def delete(namespace: str, request: Request, response: Response):
    try:
        deletion_res = await client.delete_collection(
            collection_name=namespace
        )
        print(("deletion_res", deletion_res))
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

    return DeleteNamespaceResponse(
        success=True
    )
