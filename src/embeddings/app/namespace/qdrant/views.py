from fastapi import APIRouter
from fastapi import Request
from fastapi import Response
from fastapi import HTTPException
from fastapi import status
from fastapi import Depends

from .models import NamespaceCreate
from .models import NamespaceRead
from .models import NamespaceDelete
from .models import NamespaceQuery

from embeddings.app.document.models import DocumentPagination
from embeddings.app.document.models import DocumentRead

from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from embeddings.app.config import settings

from embeddings.app.lib.cloudflare.async_api import aembed as cloudflare_aembed
from embeddings.app.lib.cloudflare.api import CloudflareEmbeddingModels

from embeddings.app.deps.request_params import CommonParams
from embeddings.app.permissions.auth import PermissionDependency

from .service import namespace as get_namespace

router = APIRouter(prefix="/namespace/qdrant")

client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT)


@router.post("", response_model=NamespaceRead)
async def create(data_in: NamespaceCreate, request: Request, response: Response):
    result = await client.create_collection(
        collection_name=data_in.name,
        vectors_config=VectorParams(size=data_in.dimensionality, distance=data_in.distance),
    )
    return await get_namespace(name=data_in.name)


@router.post("/{namespace}/query", response_model=DocumentPagination)
async def query(namespace: str, payload: NamespaceQuery, common: CommonParams, request: Request, response: Response):
    embedded_query_result = await cloudflare_aembed(model=CloudflareEmbeddingModels.BAAIBase.value, text=[payload.inputs])
    query_vectors = embedded_query_result.get('result', {}).get('data', [])
    query_vector = query_vectors[0]
    query_search_result = await client.search(
        collection_name=namespace,
        query_vector=query_vector,
        offset=common.get("offset"),
        limit=common.get("limit")
    )
    data = {
        "items": [DocumentRead(id=o.id, payload=o.payload, score=o.score, vector=o.vector) for o in query_search_result],
        "total": len(query_search_result),
        "page": common.get("page"),
    }
    return DocumentPagination(**data)


@router.get("/{namespace}", response_model=NamespaceRead, dependencies=[Depends(PermissionDependency([]))])
async def get(namespace: str, request: Request, response: Response):
    try:
        return await get_namespace(namespace)
    except UnexpectedResponse as ex:
        if ex.status_code == status.HTTP_404_NOT_FOUND:
            response.status_code = status.HTTP_404_NOT_FOUND
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=[{"msg": f"Collection with name {namespace} not found"}]
            )

        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=[{"msg": ex.content.decode('utf-8')}]
        )


@router.delete("/{namespace}", response_model=NamespaceDelete)
async def delete(namespace: str, request: Request, response: Response):
    try:
        deletion_res = await client.delete_collection(
            collection_name=namespace
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
