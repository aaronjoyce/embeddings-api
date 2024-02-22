from fastapi import APIRouter
from fastapi import Request
from fastapi import Response

from .models import CreateNamespaceResponse
from .models import CreateNamespacePayload
from .models import RetrieveNamespaceResponse


router = APIRouter(prefix="/namespace")


@router.post("", response_model=CreateNamespaceResponse)
async def create(data_in: CreateNamespacePayload, request: Request, response: Response):
    return CreateNamespaceResponse(
        success=True
    )


@router.get("/{namespace}", response_model=CreateNamespaceResponse)
async def get(namespace: str, request: Request, response: Response):
    return CreateNamespaceResponse(
        name=namespace
    )
