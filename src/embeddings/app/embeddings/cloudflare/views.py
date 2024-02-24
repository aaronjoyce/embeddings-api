from fastapi import APIRouter, Request, Response

from ..models import EmbeddingRead
from ..models import EmbeddingsCreate

from embeddings.models import InsertionResult
from embeddings.app.config import settings
from embeddings.app.lib.cloudflare.api import API, CloudflareEmbeddingModels

from .service import insert_vectors

router = APIRouter(prefix="/embeddings/cloudflare")

client = API(
    api_token=settings.CLOUDFLARE_API_TOKEN,
    account_id=settings.CLOUDFLARE_API_ACCOUNT_ID
)


@router.post("/{namespace}", response_model=InsertionResult[EmbeddingRead])
async def create(namespace: str, data_in: EmbeddingsCreate, request: Request, response: Response):
    result = client.embed(
        model=CloudflareEmbeddingModels.BAAIBase.value,
        texts=data_in.text
    )
    return insert_vectors(
        client=client,
        vectors=result.get("data", []),
        namespace=namespace,
        data_in=data_in
    )
