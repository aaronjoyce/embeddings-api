from fastapi import APIRouter
from fastapi import Depends

from .models import NamespaceCreate
from .models import NamespaceRead
from .models import NamespaceDelete
from ..models import NamespaceQuery, NamespacePagination

from embeddings.app.document.models import DocumentPagination

from embeddings.app.deps.request_params import CommonParams
from embeddings.app.permissions.auth import PermissionDependency

from .service import namespace as get
from .service import namespaces as get_all
from .service import create
from .service import delete
from .service import query


router = APIRouter(prefix="/namespace/qdrant")


@router.post("", response_model=NamespaceRead)
async def create_namespace(data_in: NamespaceCreate, ):
    return await create(data_in=data_in)


@router.post("/{namespace}/query", response_model=DocumentPagination)
async def query_namespace(namespace: str, data_in: NamespaceQuery, common: CommonParams, ):
    return await query(
        namespace=namespace,
        data_in=data_in,
        common=common
    )


@router.get("/{namespace}", response_model=NamespaceRead, dependencies=[Depends(PermissionDependency([]))])
async def get_namespace(namespace: str):
    return await get(name=namespace)


@router.get("", response_model=NamespacePagination)
async def get_namespaces():
    return await get_all()


@router.delete("/{namespace}", response_model=NamespaceDelete)
async def delete_namespace(namespace: str, ):
    return await delete(namespace)
