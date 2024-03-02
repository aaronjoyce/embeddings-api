import CloudFlare

from typing import List, Dict, Any

from app.lib.cloudflare.api import API
from app.lib.cloudflare.models import VectorPayloadItem, CreateDatabaseRecord

from app.embeddings.models import EmbeddingRead, EmbeddingCreateMulti
from app.models import InsertionResult

from app.embeddings.utils import merge_metadata, source_key
from app.exceptions import UnknownThirdPartyException, NotFoundException


from app.config import settings


def delete(client: API, namespace: str, embedding_ids: List[str]) -> Dict[str, Any]:
    try:
        return client.delete_vectors_by_ids(
            vector_index_name=namespace,
            ids=embedding_ids
        )
    except CloudFlare.exceptions.CloudFlareAPIError as ex:
        raise UnknownThirdPartyException(str(ex))


def get(client: API, namespace: str, embedding_ids: List[str]) -> List[EmbeddingRead]:
    vector_results = client.vectors_by_ids(
        vector_index_name=namespace,
        ids=embedding_ids
    )
    if len(vector_results) != len(embedding_ids):
        not_found_ids = set(embedding_ids) - set(o.get("id") for o in vector_results)
        raise NotFoundException(
            f"vectors with ids {not_found_ids} not found in the {namespace} namespace"
        )

    return [EmbeddingRead(
        id=o.get("id"),
        vector=o.get('values'),
        payload=o.get('metadata'),
        source=o.get('metadata', {}).pop(source_key(), None)
    ) for o in vector_results]


def insert(
    client: API,
    namespace: str,
    data_in: EmbeddingCreateMulti,
    vectors: List[List[float]]
) -> InsertionResult[EmbeddingRead]:
    vectors = [VectorPayloadItem(**{
        "values": vector,
        "id": meta.id,
        "metadata": merge_metadata(meta.payload, meta.text) if meta.persist_original else meta.payload
    }) for vector, meta in zip(vectors, data_in.inputs)]
    try:
        result = client.insert_vectors(
            vector_index_name=namespace,
            vectors=vectors,
            create_on_not_found=data_in.create_namespace,
            model_name=data_in.embedding_model
        )

        # writing to D1 is optional
        if settings.CLOUDFLARE_D1_DATABASE_IDENTIFIER is not None:
            insertion_records = [CreateDatabaseRecord(
                vector_id=vector_id,
                source=meta.text
            ) for vector_id, meta in zip(result.get('ids', []), data_in.inputs) if meta.persist_original]
            # conditional, as the user can optionally not persist the source text from which
            # the embedding is derived
            if insertion_records:
                insertion_result = client.upsert_database_table_records(
                    database_id=settings.CLOUDFLARE_D1_DATABASE_IDENTIFIER,
                    table_name=namespace,
                    records=insertion_records
                )
    except CloudFlare.exceptions.CloudFlareAPIError as ex:
        raise UnknownThirdPartyException(
            str(ex)
        )

    items = [EmbeddingRead(id=o) for o in result.get('ids', [])]
    return InsertionResult[EmbeddingRead](
        count=len(items),
        items=items
    )
