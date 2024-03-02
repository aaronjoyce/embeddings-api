import CloudFlare

from typing import List, Dict, Any

from .models import NamespaceCreate, NamespaceRead, NamespaceDelete
from ..models import NamespaceQuery, NamespacePagination, NamespaceBaseModel

from embeddings.app.lib.cloudflare.api import API
from embeddings.app.embeddings.utils import source_key
from embeddings.app.document.models import DocumentRead, DocumentPagination
from embeddings.app.lib.cloudflare.api import CloudflareEmbeddingModels, ERROR_CODE_VECTOR_INDEX_NOT_FOUND

from embeddings.app.config import settings

from embeddings.app.deps.request_params import CommonParams
from embeddings.exceptions import NotFoundException, UnknownThirdPartyException


def create(client: API, data_in: NamespaceCreate) -> NamespaceRead:
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
        raise NotFoundException(
            f"Cloudflare vectors with ids {ids} not found in the '{namespace}' vectorize index."
        )

    results = query_result[0].get('results', [])
    if len(results) != len(ids):
        not_found_ids = set(ids) - set(o.get("id") for o in results)
        raise NotFoundException(
            f"Cloudflare vectors with ids {not_found_ids} not found in the '{namespace}' vectorize index."
        )

    return results


def paginated_query_results(matches: List, common: CommonParams) -> DocumentPagination:
    data = {
        "items": [
            DocumentRead(
                id=vector.get('id'),
                payload=vector.get('metadata'),
                score=vector.get('score'),
                vector=vector.get('values'),
                source=vector.get('metadata').pop(source_key(), None)
            ) for vector in matches
        ],
        "total": len(matches),
        "page": common.get("page"),
        "itemsPerPage": common.get("limit")
    }
    return DocumentPagination(**data)


def vector_indexes(client: API) -> NamespacePagination:
    try:
        res = client.list_vector_indexes()
        return NamespacePagination(
            items=[
                NamespaceBaseModel(
                    name=o.get('name')
                ) for o in res
            ],
            total=len(res),
            page=1,
            itemsPerPage=len(res)
        )
    except CloudFlare.exceptions.CloudFlareAPIError as ex:
        raise UnknownThirdPartyException(str(ex))


def vector_index_by_name(client: API, namespace: str, ) -> NamespaceRead:
    try:
        res = client.vector_index_by_name(
            namespace
        )
    except CloudFlare.exceptions.CloudFlareAPIError as ex:
        if int(ex) == ERROR_CODE_VECTOR_INDEX_NOT_FOUND:
            raise NotFoundException(
                f"Vector index with name '{namespace}' not found."
            )
        else:
            raise NotFoundException(str(ex))

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
        raise UnknownThirdPartyException(
            str(ex)
        )

    return NamespaceDelete(
        success=True
    )
