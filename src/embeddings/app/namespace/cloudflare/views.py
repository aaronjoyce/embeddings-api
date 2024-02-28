from fastapi import APIRouter, Response, Request, Depends

from .models import (
    NamespaceCreate,
    NamespaceRead,
    NamespaceDelete,
)

from ..models import NamespaceQuery

from embeddings.app.permissions.auth import PermissionDependency, DefaultPermission
from embeddings.app.document.models import DocumentPagination
from embeddings.app.config import settings
from embeddings.app.deps.request_params import CommonParams
from embeddings.app.lib.cloudflare.api import API

from .service import (
    create_vector_index,
    embedding_matches,
    vectors_by_ids,
    paginated_query_results,
    vector_index_by_name,
    delete_vector_index_by_name
)


router = APIRouter(prefix="/namespace/cloudflare")

client = API(
    api_token=settings.CLOUDFLARE_API_TOKEN,
    account_id=settings.CLOUDFLARE_API_ACCOUNT_ID
)


@router.post("", response_model=NamespaceRead)
async def create(data_in: NamespaceCreate, ):
    return create_vector_index(client=client, data_in=data_in)


@router.post("/{namespace}/query", response_model=DocumentPagination)
async def query(namespace: str, data_in: NamespaceQuery, common: CommonParams, ):
    matches = embedding_matches(client=client, namespace=namespace, data_in=data_in)
    results = vectors_by_ids(client=client, namespace=namespace, ids=[o.get("id") for o in matches])
    return paginated_query_results(
        results=results,
        matches=matches,
        common=common
    )


@router.get(
    "/{namespace}",
    response_model=NamespaceRead,
    dependencies=[Depends(PermissionDependency([DefaultPermission]))]
)
async def get(namespace: str):
    return vector_index_by_name(client=client, namespace=namespace)


@router.delete("/{namespace}", response_model=NamespaceDelete)
async def delete(namespace: str, ):
    return delete_vector_index_by_name(client=client, namespace=namespace)
