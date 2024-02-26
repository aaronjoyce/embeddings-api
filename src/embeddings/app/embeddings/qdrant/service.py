import re

from fastapi import status, HTTPException
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.conversions import common_types
from qdrant_client.http.models import UpdateStatus

from ..models import EmbeddingRead, EmbeddingPagination, EmbeddingCreateMulti

from embeddings.app.lib.cloudflare.api import API, DIMENSIONALITY_PRESETS
from embeddings.app.lib.cloudflare.api import CloudflareEmbeddingModels

from embeddings.app.lib.cloudflare.models import CreateDatabaseRecord

from embeddings.app.deps.request_params import CommonParams
from embeddings.models import InsertionResult

from embeddings.app.config import settings


cloudflare = API(
    api_token=settings.CLOUDFLARE_API_TOKEN,
    account_id=settings.CLOUDFLARE_API_ACCOUNT_ID
)


async def embedding(client: AsyncQdrantClient, namespace: str, embedding_id: str):
    try:
        result = await client.retrieve(
            collection_name=namespace,
            ids=[embedding_id],
            with_vectors=False,
            with_payload=True
        )
        return EmbeddingRead(
            id=result[0].id
        )
    except UnexpectedResponse as ex:
        if ex.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=[{"msg": ex.content.decode('utf-8')}]
            )


async def embeddings(client: AsyncQdrantClient, namespace: str, common: CommonParams):
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


async def collection_exists(client: AsyncQdrantClient, namespace: str) -> bool:
    try:
        # check if the collection exists
        await client.get_collection(
            collection_name=namespace
        )
    except UnexpectedResponse as ex:
        return False
    return True


async def insert_embedding(
        client: AsyncQdrantClient,
        namespace: str,
        data_in: EmbeddingCreateMulti,
) -> InsertionResult:
    texts = [o.text for o in data_in.inputs]
    result = cloudflare.embed(
        model=str(data_in.embedding_model),
        texts=texts
    )
    try:
        upsert_result = await client.upsert(
            collection_name=namespace,
            points=[common_types.PointStruct(**{
                "vector": vector,
                "id": meta.id,
                "payload": meta.payload
            }) for vector, meta in zip(result.get('data', []), data_in.inputs)]
        )
        if upsert_result.status != UpdateStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=[{"msg": "Error occurred whilst attempting to upsert data in Qdrant"}]
            )
    except UnexpectedResponse as ex:
        if ex.status_code == status.HTTP_400_BAD_REQUEST:
            expected_dimension_error = re.search(r"expected dim: (\d+), got (\d+)", str(ex))
            if expected_dimension_error:
                expected_dimension = int(expected_dimension_error.group(1))
                received_dimension = int(expected_dimension_error.group(2))
                compatible_model_names = ','.join([str(o) for o in DIMENSIONALITY_PRESETS.get(expected_dimension, [])])
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=[{
                        "msg": f"The embedding model's dimensionality: {received_dimension} "
                               f"(defaults to {data_in.embedding_model}) is not compatible with the "
                               f"dimensionality of the namespace '{namespace}', dimensionality: {expected_dimension}. "
                               f"Please provide one of the following compatible models: {compatible_model_names}",
                    }]
                )

        raise ex

    insertion_records = [CreateDatabaseRecord(
        vector_id=o.id,
        source=o.text
    ) for o in data_in.inputs]
    insertion_result = cloudflare.upsert_database_table_records(
        database_id=settings.CLOUDFLARE_D1_DATABASE_IDENTIFIER,
        table_name=namespace,
        records=insertion_records
    )
    if not insertion_result.get('success'):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=[{"msg": "Something went wrong whilst attempting to persist the source text to Cloudflare D1"}]
        )

    items = [EmbeddingRead(id=o.vector_id) for o in insertion_records]
    return InsertionResult[EmbeddingRead](
        count=len(items),
        items=items
    )
