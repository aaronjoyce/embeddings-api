from fastapi import APIRouter
from fastapi import Request
from fastapi import Response

from .models import GetEmbeddingsResponse
from .models import CreateEmbeddingsResponse
from .models import CreateEmbeddingsDataIn

from .service import create_cloudflare_embedding

router = APIRouter(prefix="/embeddings")


@router.get("", response_model=GetEmbeddingsResponse)
async def get(request: Request, response: Response):
    return GetEmbeddingsResponse()


@router.post("/{namespace}", response_model=CreateEmbeddingsResponse)
async def post(namespace: str, data_in: CreateEmbeddingsDataIn, request: Request, response: Response):
    print(("namespace", namespace))
    print(("data_in", data_in, ))
    res = create_cloudflare_embedding(
        model="@cf/baai/bge-base-en-v1.5",
        text=data_in.text
    )
    print(("res", res))
    return CreateEmbeddingsResponse()
