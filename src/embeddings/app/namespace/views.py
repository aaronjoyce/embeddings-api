from fastapi import APIRouter
from fastapi import Request
from fastapi import Response

from pydantic import BaseModel
from pydantic import Field


router = APIRouter(prefix="/namespace")


class CreateNamespacePayload(BaseModel):
    name: str
    dimensions: int = Field(default=1024)


class CreateNamespaceResponse(BaseModel):
    success: bool


class RetrieveNamespaceResponse(BaseModel):
    name: str


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
