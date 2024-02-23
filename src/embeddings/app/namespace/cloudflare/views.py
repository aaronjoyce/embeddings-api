import CloudFlare

from fastapi import APIRouter
from fastapi import Request
from fastapi import Response
from fastapi import HTTPException
from fastapi import status

from http import HTTPStatus

from .models import NamespaceCreate
from .models import NamespaceRead
from .models import NamespaceDelete
from .models import NamespaceQuery

from embeddings.app.document.models import DocumentPagination
from embeddings.app.document.models import DocumentRead

from embeddings.app.config import settings

from embeddings.app.lib.cloudflare.api import CloudflareEmbeddingModels

from embeddings.app.deps.request_params import CommonParams


from embeddings.app.lib.cloudflare.api import ERROR_CODE_VECTOR_INDEX_NOT_FOUND
from embeddings.app.lib.cloudflare.api import API


router = APIRouter(prefix="/namespace/cloudflare")

client = API(
    api_token=settings.CLOUDFLARE_MASTER_API_TOKEN,
    account_id=settings.CLOUDFLARE_API_ACCOUNT_ID
)


@router.post("", response_model=NamespaceRead)
async def create(data_in: NamespaceCreate, request: Request, response: Response):
    res = client.create_vector_index(
        name=data_in.name,
        preset=data_in.preset
    )
    print(("res", res))
    return NamespaceRead(
        name=res.result.name,
        dimensionality=res.result.config.dimensions,
        metric=res.result.metric.lower()
    )


@router.post("/{namespace}/query", response_model=DocumentPagination)
async def query(namespace: str, payload: NamespaceQuery, common: CommonParams, request: Request, response: Response):
    res = client.embed(
        model=CloudflareEmbeddingModels.BAAIBase.value,
        texts=[payload.inputs]
    )
    query_vectors = res.get('result', {}).get('data', [])
    query_vector = query_vectors[0]
    query_search_result = client.query_vector_index(
        vector_index_name=namespace,
        vector=query_vector
    )
    matches = query_search_result.matches
    data = {
        "items": [DocumentRead(id=o.id, payload=o.metadata, score=o.score, vector=o.values) for o in matches],
        "total": len(query_search_result),
        "page": common.get("page"),
    }
    return DocumentPagination(**data)


@router.get("/{namespace}", response_model=NamespaceRead)
async def get(namespace: str, request: Request, response: Response):
    try:
        res = client.vector_index_by_name(
            namespace
        )
        print(("vector_index.res", res))
    except CloudFlare.exceptions.CloudFlareAPIError as ex:
        if int(ex) == ERROR_CODE_VECTOR_INDEX_NOT_FOUND:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=[{
                    "msg": f"Vector index with name '{namespace}' not found."
                }]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=[{"msg": str(ex)}]
            )

    return NamespaceRead(
        name=res.name,
        dimensionality=res.config.dimensions,
        metric=res.config.metric.lower()
    )


@router.delete("/{namespace}", response_model=NamespaceDelete)
async def delete(namespace: str, request: Request, response: Response):
    try:
        deletion_res = await client.delete_vector_index_by_name(
            name=namespace
        )
        print(("deletion_res", deletion_res))
    except CloudFlare.exceptions.CloudFlareAPIError as ex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=[{"msg": str(ex)}]
        )

    return NamespaceDelete(
        success=True
    )
