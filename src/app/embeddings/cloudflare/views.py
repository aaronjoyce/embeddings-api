from fastapi import APIRouter, status

from ..models import EmbeddingRead
from ..models import EmbeddingCreateMulti
from ..models import EmbeddingPagination

from app.models import InsertionResult
from app.config import settings

from .service import insert, get, delete

from ..models import EmbeddingDelete

from app.deps.request_params import CommonParams
from app.deps.cloudflare import CloudflareClient
from app.embeddings.utils import source_key

from app.exceptions import EnvironmentVariableConfigException


router = APIRouter(prefix="/embeddings/cloudflare")


@router.get("/{namespace}", response_model=EmbeddingPagination)
async def get_embeddings(namespace: str, common: CommonParams, client: CloudflareClient):
    """
    Page through embeddings.
    Only supported if a valid `CLOUDFLARE_D1_DATABASE_IDENTIFIER` environment variable has been set.
    """
    if settings.CLOUDFLARE_D1_DATABASE_IDENTIFIER is None:
        raise EnvironmentVariableConfigException(
            "Support for listing embeddings is unavailable without integrating Cloudflare D1."
        )

    response = client.list_database_table_records(
        database_id=settings.CLOUDFLARE_D1_DATABASE_IDENTIFIER,
        table_name=namespace,
        limit=common.get("limit"),
        offset=common.get("offset")
    )
    d1_results = {o.get('vector_id'): o for o in response[0].get('results', [])}

    response = client.vectors_by_ids(
        vector_index_name=namespace,
        ids=list(d1_results.keys())
    )
    vectorize_results = {o.get('id'): o for o in response}
    return EmbeddingPagination(
        items=[EmbeddingRead(
            id=vector_id,
            source=vectorize_results.get(vector_id, {}).get('metadata', {}).pop(source_key(), None),
            vector=vectorize_results.get(vector_id, {}).get('vector'),
            payload=vectorize_results.get(vector_id, {}).get('metadata', {})
        ) for vector_id, record in d1_results],
        itemsPerPage=len(vectorize_results),
        page=common.get("page")
    )


@router.get("/{namespace}/{embedding_id}", response_model=EmbeddingRead)
async def get_embedding(namespace: str, embedding_id: str, client: CloudflareClient):
    """Retrieve a single embedding vector by namespace and embedding `ID`"""
    return get(
        client=client,
        namespace=namespace,
        embedding_ids=[embedding_id]
    )[0]


@router.delete("/{namespace}/{embedding_id}", response_model=EmbeddingDelete)
async def delete_embedding(namespace: str, embedding_id: str, client: CloudflareClient):
    """Delete an existing embedding by namespace and `ID`"""
    result = delete(
        client=client,
        namespace=namespace,
        embedding_ids=[embedding_id]
    )
    return EmbeddingDelete(
        success=True,
        count=result.get('count')
    )


@router.post("/{namespace}", response_model=InsertionResult[EmbeddingRead], status_code=status.HTTP_201_CREATED)
async def create_embedding(namespace: str, data_in: EmbeddingCreateMulti, client: CloudflareClient):
    """
    Generate and persist embeddings for one or more text items.

    If a valid `CLOUDFLARE_D1_DATABASE_IDENTIFIER` environment variable has been set,
    the embedding identifier, along with the original embedded text, will be persisted
    to the Cloudflare D1 service.

    Persisting to Cloudflare D1 enables API support for paging through embeddings, i.e.,
    `/embeddings/cloudflare/{namespace}`. This is because the Cloudflare the Vectorize
    service does not natively support paging/scrolling through vectors at this time.
    """
    result = client.embed(
        model=data_in.embedding_model.value,
        texts=[o.text for o in data_in.inputs]
    )
    return insert(
        client=client,
        vectors=result.get("data", []),
        namespace=namespace,
        data_in=data_in
    )
