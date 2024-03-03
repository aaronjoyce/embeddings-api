
from fastapi import APIRouter, Depends, status

from .models import (
    NamespaceCreate,
    NamespaceRead,
    NamespaceDelete,
)

from ..models import NamespaceQuery, NamespacePagination

from app.permissions.auth import PermissionDependency, DefaultPermission
from app.document.models import DocumentPagination
from app.deps.request_params import CommonParams
from app.deps.cloudflare import CloudflareClient

from .service import (
    create,
    embedding_matches,
    paginated_query_results,
    vector_index_by_name,
    delete_vector_index_by_name,
    vector_indexes
)


router = APIRouter(prefix="/namespace/cloudflare")


@router.post("", response_model=NamespaceRead, status_code=status.HTTP_201_CREATED)
async def create_namespace(data_in: NamespaceCreate, client: CloudflareClient):
    """Create a Cloudflare vector index"""
    return create(client=client, data_in=data_in)


@router.get("", response_model=NamespacePagination)
async def get_namespaces(client: CloudflareClient):
    """Retrieve all Cloudflare vector indexes."""
    return vector_indexes(client=client)


@router.post("/{namespace}/query", response_model=DocumentPagination)
async def query_namespace(namespace: str, data_in: NamespaceQuery, common: CommonParams, client: CloudflareClient):
    """Run a vector query against a named vector index."""
    matches = embedding_matches(client=client, namespace=namespace, data_in=data_in)
    return paginated_query_results(
        matches=matches,
        common=common
    )


@router.get(
    "/{namespace}",
    response_model=NamespaceRead,
    dependencies=[Depends(PermissionDependency([DefaultPermission]))]
)
async def get_namespace(namespace: str, client: CloudflareClient):
    """Retrieve a vector index by name."""
    return vector_index_by_name(client=client, namespace=namespace)


@router.delete("/{namespace}", response_model=NamespaceDelete)
async def delete_namespace(namespace: str, client: CloudflareClient):
    """Delete a vector index by name."""
    return delete_vector_index_by_name(client=client, namespace=namespace)
