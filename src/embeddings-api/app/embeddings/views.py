from fastapi import APIRouter
from fastapi import Request
from fastapi import Response

from .models import GetEmbeddingsResponse
from .models import CreateEmbeddingsResponse

router = APIRouter(prefix="/embeddings")


@router.get("", response_model=GetEmbeddingsResponse)
async def get(request: Request, response: Response):
    return GetEmbeddingsResponse()


@router.post("/{namespace}", response_model=CreateEmbeddingsResponse)
async def post(namespace: str, request: Request, response: Response):
    print(("namespace", namespace))
    return CreateEmbeddingsResponse()
