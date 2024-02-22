import uuid
from enum import Enum

from fastapi import status
from fastapi import APIRouter
from fastapi import Request
from fastapi import Response
from fastapi import HTTPException

from .models import EmbeddingRead
from .models import EmbeddingsCreate
from .models import EmbeddingPagination

from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.conversions import common_types
from qdrant_client.models import Filter

from embeddings.app.deps.request_params import CommonParams


from embeddings.app.lib.cloudflare import embed as cloudflare_embed

from embeddings.app.config import CloudflareEmbeddingModels
from embeddings.app.config import settings

router = APIRouter(prefix="/embeddings")

client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT)


@router.get("/{namespace}/{embedding_id}", response_model=EmbeddingRead)
async def get_embedding(namespace: str, embedding_id: str, request: Request, response: Response):
    try:
        result = await client.retrieve(
            collection_name=namespace,
            ids=[embedding_id],
            with_vectors=False,
            with_payload=True
        )
        point = result[0]
        return EmbeddingRead(
            id=point.id
        )
    except UnexpectedResponse as ex:
        if ex.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=[{"msg": ex.content.decode('utf-8')}]
            )


@router.get("/{namespace}", response_model=EmbeddingPagination)
async def list_embeddings(namespace: str, common: CommonParams, request: Request, response: Response):
    points, offset = await client.scroll(
        collection_name=namespace,
        limit=common.get("limit"),
        offset=common.get("offset")
    )
    body = {
        "total": len(points),
        "page": common.get("page"),
        "items": [{"id": o.id} for o in points]
    }
    return EmbeddingPagination(**body)


@router.post("/{namespace}", response_model=EmbeddingRead)
async def post(namespace: str, data_in: EmbeddingsCreate, request: Request, response: Response):
    try:
        # check if the collection exists
        await client.get_collection(
            collection_name=namespace
        )
    except UnexpectedResponse as ex:
        if ex.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(
                status_code=404,
                detail=[{"msg": f"Collection with name {namespace} not found"}]
            )

        raise HTTPException(
            status_code=500,
            detail=[{"msg": ex.content.decode('utf-8')}]
        )

    result = cloudflare_embed(
        model=CloudflareEmbeddingModels.BAAIBase.value,
        text=data_in.text
    )
    embedding_vectors = result.get('result', {}).get('data', [])
    if not embedding_vectors:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=[{"msg": "Empty embedding vector response from the Cloudflare API"}]
        )

    new_id = str(uuid.uuid4())
    point_data = {
        "id": new_id,
        "vector": embedding_vectors[0]
    }
    if data_in.payload:
        point_data["payload"] = data_in.payload

    upsert_result = await client.upsert(
        collection_name=namespace,
        points=[common_types.PointStruct(**point_data)]
    )
    return EmbeddingRead(
        id=new_id,
    )

