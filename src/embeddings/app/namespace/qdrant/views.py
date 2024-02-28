from fastapi import APIRouter
from fastapi import Request
from fastapi import Response
from fastapi import HTTPException
from fastapi import status
from fastapi import Depends

from .models import NamespaceCreate
from .models import NamespaceRead
from .models import NamespaceDelete
from ..models import NamespaceQuery

from embeddings.app.document.models import DocumentPagination

from embeddings.app.config import settings

from embeddings.app.deps.request_params import CommonParams
from embeddings.app.permissions.auth import PermissionDependency

from .service import namespace as get_namespace
from .service import create as create_namespace
from .service import delete as delete_namespace
from .service import embedding_matches


router = APIRouter(prefix="/namespace/qdrant")


@router.post("", response_model=NamespaceRead)
async def create(data_in: NamespaceCreate, ):
    return await create_namespace(data_in=data_in)


@router.post("/{namespace}/query", response_model=DocumentPagination)
async def query(namespace: str, data_in: NamespaceQuery, common: CommonParams, ):
    return await embedding_matches(
        namespace=namespace,
        data_in=data_in,
        common=common
    )


@router.get("/{namespace}", response_model=NamespaceRead, dependencies=[Depends(PermissionDependency([]))])
async def get(namespace: str):
    return await get_namespace(name=namespace)


@router.delete("/{namespace}", response_model=NamespaceDelete)
async def delete(namespace: str, ):
    return await delete_namespace(namespace)
