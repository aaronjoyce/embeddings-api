import re

from typing import List

from fastapi import status
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.conversions import common_types
from qdrant_client.http.models import UpdateStatus
from qdrant_client.models import PointIdsList
from qdrant_client.http.models import Distance, VectorParams

from ..models import EmbeddingRead, EmbeddingPagination, EmbeddingCreateMulti, EmbeddingDelete

from app.lib.cloudflare.api import API, DIMENSIONALITY_PRESETS
from app.embeddings.utils import source_key
from app.embeddings.utils import merge_metadata
from app.exceptions import NotFoundException, UnknownThirdPartyException, EmbeddingDimensionalityException

from app.lib.cloudflare.models import CreateDatabaseRecord

from app.deps.request_params import CommonParams
from app.models import InsertionResult

from app.config import settings


cloudflare = API(
    api_token=settings.CLOUDFLARE_API_TOKEN,
    account_id=settings.CLOUDFLARE_API_ACCOUNT_ID
)


async def embedding(client: AsyncQdrantClient, namespace: str, embedding_id: str):
    try:
        result = await client.retrieve(
            collection_name=namespace,
            ids=[embedding_id],
            with_vectors=True,
            with_payload=True
        )
        return EmbeddingRead(
            id=result[0].id,
            payload=result[0].payload,
            vector=result[0].vector,
            source=result[0].payload.pop(source_key(), None)
        )
    except UnexpectedResponse as ex:
        if ex.status_code == status.HTTP_404_NOT_FOUND:
            raise NotFoundException(
                ex.content.decode('utf-8')
            )
        else:
            raise UnknownThirdPartyException(
                ex.content.decode('utf-8')
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


async def create(client: AsyncQdrantClient, namespace: str, data_in: EmbeddingCreateMulti) -> InsertionResult:
    exists = await collection_exists(client, namespace)
    if not exists and not data_in.create_namespace:
        raise NotFoundException(
            f"Collection with name {namespace} does not exist"
        )

    if not exists:
        vector_size = data_in.embedding_model.dimensionality
        await client.create_collection(
            collection_name=namespace,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ),
        )

    return await insert(
        client=client,
        data_in=data_in,
        namespace=namespace,
    )


async def insert(
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
                "payload": merge_metadata(meta.payload, meta.text) if meta.persist_original else meta.payload
            }) for vector, meta in zip(result.get('data', []), data_in.inputs)]
        )
        if upsert_result.status != UpdateStatus.COMPLETED:
            raise UnknownThirdPartyException(
                "Error occurred whilst attempting to upsert data in Qdrant"
            )
    except UnexpectedResponse as ex:
        if ex.status_code == status.HTTP_400_BAD_REQUEST:
            expected_dimension_error = re.search(r"expected dim: (\d+), got (\d+)", str(ex))
            if expected_dimension_error:
                expected_dimension = int(expected_dimension_error.group(1))
                received_dimension = int(expected_dimension_error.group(2))
                compatible_model_names = ','.join([str(o) for o in DIMENSIONALITY_PRESETS.get(expected_dimension, [])])
                raise EmbeddingDimensionalityException(
                    f"The embedding model's dimensionality: {received_dimension} "
                    f"(defaults to {data_in.embedding_model}) is not compatible with the "
                    f"dimensionality of the namespace '{namespace}', dimensionality: {expected_dimension}. "
                    f"Please provide one of the following compatible models: {compatible_model_names}",
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
        raise UnknownThirdPartyException(
            "Something went wrong whilst attempting to persist the source text to Cloudflare D1"
        )

    items = [EmbeddingRead(id=o.vector_id) for o in insertion_records]
    return InsertionResult[EmbeddingRead](
        count=len(items),
        items=items
    )


async def delete(client: AsyncQdrantClient, namespace: str, embedding_ids: List[str]) -> EmbeddingDelete:
    response = await client.delete(
        collection_name=namespace,
        points_selector=PointIdsList(
            points=embedding_ids
        )
    )
    success = response.status == UpdateStatus.COMPLETED
    return EmbeddingDelete(
        success=success,
        count=None
    )
