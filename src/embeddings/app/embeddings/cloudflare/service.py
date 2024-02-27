import CloudFlare

from typing import List

from fastapi import HTTPException, status

from embeddings.app.lib.cloudflare.api import API
from embeddings.app.lib.cloudflare.api import ERROR_CODE_VECTOR_INDEX_NOT_FOUND
from embeddings.app.lib.cloudflare.models import VectorPayloadItem, CreateDatabaseRecord

from embeddings.app.embeddings.models import EmbeddingRead, EmbeddingCreateMulti
from embeddings.models import InsertionResult

from embeddings.app.config import settings


def delete_embeddings(client: API, namespace: str, embedding_ids: List[str]) -> List[str]:
    return client.delete_vectors_by_ids(
        vector_index_name=namespace,
        ids=embedding_ids
    )


def get_embeddings(client: API, namespace: str, embedding_ids: List[str]) -> List[EmbeddingRead]:
    vector_results = client.vectors_by_ids(
        vector_index_name=namespace,
        ids=embedding_ids
    )
    if len(vector_results) != len(embedding_ids):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND
        )

    response = client.database_table_records_by_vector_ids(
        database_id=settings.CLOUDFLARE_D1_DATABASE_IDENTIFIER,
        table_name=namespace,
        vector_ids=embedding_ids
    )
    d1_results = {o.get('vector_id'): o for o in response[0].get('results', [])}

    return [EmbeddingRead(
        id=o.get("id"),
        vector=o.get('values'),
        payload=o.get('metadata'),
        source=d1_results.get(o.get("id")).get('source')
    ) for o in vector_results]


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
            create_on_not_found=data_in.create_namespace,
            model_name=data_in.embedding_model
        )
        insertion_records = [CreateDatabaseRecord(
            vector_id=vector_id,
            source=meta.text
        ) for vector_id, meta in zip(result.get('ids', []), data_in.inputs) if meta.persist_source]
        # conditional, as the user can optionally not persist the source text from which
        # the embedding is derived
        if insertion_records:
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
                    f"Create the index via a separate call or include 'create_namespace' "
                    f"in your payload to automagically create and insert."
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
