from fastapi import APIRouter
from fastapi import Depends
from fastapi import status

from .models import NamespaceCreate
from .models import NamespaceRead
from .models import NamespaceDelete
from ..models import NamespaceQuery, NamespacePagination

from app.document.models import DocumentPagination

from app.deps.request_params import CommonParams
from app.deps.qdrant import QdrantClient
from app.permissions.auth import PermissionDependency

from .service import namespace as get
from .service import namespaces as get_all
from .service import create
from .service import delete
from .service import query


router = APIRouter(prefix="/namespace/qdrant")


@router.post("", response_model=NamespaceRead, status_code=status.HTTP_201_CREATED)
async def create_namespace(data_in: NamespaceCreate, client: QdrantClient):
    return await create(data_in=data_in, client=client)


@router.post("/{namespace}/query", response_model=DocumentPagination)
async def query_namespace(namespace: str, data_in: NamespaceQuery, common: CommonParams, ):
    """Run a vector query against a named collection."""
    return await query(
        namespace=namespace,
        data_in=data_in,
        common=common
    )


@router.get("/{namespace}", response_model=NamespaceRead, dependencies=[Depends(PermissionDependency([]))])
async def get_namespace(namespace: str, client: QdrantClient):
    """Retrieve a collection by name"""
    return await get(name=namespace, client=client)


@router.get("", response_model=NamespacePagination)
async def get_namespaces(client: QdrantClient):
    """Retrieve all collections"""
    return await get_all(client=client)


@router.delete("/{namespace}", response_model=NamespaceDelete)
async def delete_namespace(namespace: str, client: QdrantClient):
    """Delete a collection by name."""
    return await delete(namespace, client=client)
