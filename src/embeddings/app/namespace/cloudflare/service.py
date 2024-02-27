import CloudFlare

from typing import List, Dict, Any

from fastapi import HTTPException
from fastapi import status

from embeddings.app.lib.cloudflare.api import API

from .models import NamespaceCreate, NamespaceRead, NamespaceQuery, NamespaceDelete

from embeddings.app.document.models import DocumentRead, DocumentPagination

from embeddings.app.lib.cloudflare.api import CloudflareEmbeddingModels, ERROR_CODE_VECTOR_INDEX_NOT_FOUND

from embeddings.app.config import settings

from embeddings.app.deps.request_params import CommonParams


def create_vector_index(client: API, data_in: NamespaceCreate) -> NamespaceRead:
    res = client.create_vector_index(
        name=data_in.name,
        preset=data_in.preset
    )
    config = res.get('config', {})
    return NamespaceRead(
        name=res.get('name'),
        dimensionality=config.get('dimensions'),
        metric=config.get("metric").lower()
    )


def embedding_matches(client: API, namespace: str, data_in: NamespaceQuery):
    res = client.embed(
        model=CloudflareEmbeddingModels.BAAIBase.value,
        texts=[data_in.inputs]
    )
    query_vectors = res.get('data', [])
    query_vector = query_vectors[0]
    query_search_result = client.query_vector_index(
        vector_index_name=namespace,
        vector=query_vector,
        return_vectors=data_in.return_vectors,
        return_metadata=data_in.return_metadata,
        top_k=data_in.limit,
        metadata_filter=data_in.filter
    )
    return query_search_result.get('matches', [])


def vectors_by_ids(client: API, namespace: str, ids: List[str]) -> List[Dict[str, Any]]:
    query_result = client.database_table_records_by_vector_ids(
        database_id=settings.CLOUDFLARE_D1_DATABASE_IDENTIFIER,
        table_name=namespace,
        vector_ids=ids
    )
    if not query_result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    results = query_result[0].get('results', [])
    if len(results) != len(ids):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "msg": f"Inconsistent state between the number of results returned from Vectorize and D1. "
                       f"Got {len(results)} results from D1, expected {len(ids)}."
            }
        )

    return results


def paginated_query_results(results: List, matches: List, common: CommonParams) -> DocumentPagination:
    data = {
        "items": [
            DocumentRead(
                id=vector.get('id'),
                payload=vector.get('metadata'),
                score=vector.get('score'),
                vector=vector.get('values'),
                source=d1_record.get('source')
            ) for vector, d1_record in zip(matches, results)
        ],
        "total": len(matches),
        "page": common.get("page"),
        "itemsPerPage": common.get("limit")
    }
    return DocumentPagination(**data)


def vector_index_by_name(client: API, namespace: str, ) -> NamespaceRead:
    try:
        res = client.vector_index_by_name(
            namespace
        )
    except CloudFlare.exceptions.CloudFlareAPIError as ex:
        if int(ex) == ERROR_CODE_VECTOR_INDEX_NOT_FOUND:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=[{
                    "msg": f"Vector index with name '{namespace}' not found."
                }]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=[{"msg": str(ex)}]
            )

    return NamespaceRead(
        name=res.get('name'),
        dimensionality=res.get('config', {}).get('dimensions'),
        metric=res.get('config').get('metric').lower()
    )


def delete_vector_index_by_name(client: API, namespace: str) -> NamespaceDelete:
    try:
        deletion_res = client.delete_vector_index_by_name(
            name=namespace
        )
        # also need to check whether there's a corresponding table in d1
    except CloudFlare.exceptions.CloudFlareAPIError as ex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=[{"msg": str(ex)}]
        )

    return NamespaceDelete(
        success=True
    )
