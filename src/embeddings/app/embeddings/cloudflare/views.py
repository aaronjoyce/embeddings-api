from fastapi import APIRouter, Request, Response

from ..models import EmbeddingRead
from ..models import EmbeddingsCreate, EmbeddingCreateMulti

from embeddings.models import InsertionResult
from embeddings.app.config import settings
from embeddings.app.lib.cloudflare.api import API, CloudflareEmbeddingModels

from .service import insert_vectors, get_embeddings, delete_embeddings

from ..models import EmbeddingDelete

router = APIRouter(prefix="/embeddings/cloudflare")

client = API(
    api_token=settings.CLOUDFLARE_API_TOKEN,
    account_id=settings.CLOUDFLARE_API_ACCOUNT_ID
)


@router.get("/{namespace}/{embedding_id}", response_model=EmbeddingRead)
async def get(namespace: str, embedding_id: str, request: Request, response: Response):
    embedding_results = get_embeddings(
        client=client,
        namespace=namespace,
        embedding_ids=[embedding_id]
    )
    return embedding_results[0]


@router.delete("/{namespace}/{embedding_id}", response_model=EmbeddingDelete)
async def delete(namespace: str, embedding_id: str, request: Request, response: Response):
    result = delete_embeddings(
        client=client,
        namespace=namespace,
        embedding_ids=[embedding_id]
    )
    return EmbeddingDelete(
        success=True
    )


@router.post("/{namespace}", response_model=InsertionResult[EmbeddingRead])
async def create(namespace: str, data_in: EmbeddingCreateMulti, request: Request, response: Response):
    texts = [o.text for o in data_in.inputs]

    result = client.embed(
        model=data_in.embedding_model.value,
        texts=texts
    )
    return insert_vectors(
        client=client,
        vectors=result.get("data", []),
        namespace=namespace,
        data_in=data_in
    )
