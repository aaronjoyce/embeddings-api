import CloudFlare

from typing import List

from fastapi import HTTPException, status

from embeddings.app.lib.cloudflare.api import API
from embeddings.app.lib.cloudflare.api import ERROR_CODE_VECTOR_INDEX_NOT_FOUND
from embeddings.app.lib.cloudflare.models import VectorPayloadItem, CreateDatabaseRecord

from embeddings.app.embeddings.models import EmbeddingRead, EmbeddingCreateMulti
from embeddings.models import InsertionResult

from embeddings.app.config import settings


def insert_vectors(
    client: API,
    namespace: str,
    data_in: EmbeddingCreateMulti,
    vectors: List[List[float]]
) -> InsertionResult[EmbeddingRead]:
    vectors = [VectorPayloadItem(**{
        "values": vector,
        "id": meta.id,
        "metadata": meta.payload
    }) for vector, meta in zip(vectors, data_in.inputs)]
    try:
        result = client.insert_vectors(
            vector_index_name=namespace,
            vectors=vectors,
            create_on_not_found=data_in.create_namespace
        )
        insertion_records = [CreateDatabaseRecord(
            vector_id=vector_id,
            source=meta.text
        ) for vector_id, meta in zip(result.get('ids', []), data_in.inputs)]
        insertion_result = client.upsert_database_table_records(
            database_id=settings.CLOUDFLARE_D1_DATABASE_IDENTIFIER,
            table_name=namespace,
            records=insertion_records
        )
    except CloudFlare.exceptions.CloudFlareAPIError as ex:
        if int(ex) == ERROR_CODE_VECTOR_INDEX_NOT_FOUND:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=[{
                    "msg": f"Vector index with name '{namespace}' not found. "
                    f"Create the index via a separate call or include 'create_index' "
                    f"in your payload to automagically create and insert"
                }]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=[{"msg": str(ex)}]
            )

    items = [EmbeddingRead(id=o) for o in result.get('ids', [])]
    return InsertionResult[EmbeddingRead](
        count=len(items),
        items=items
    )
