from embeddings.app.lib.cloudflare.api import API
from embeddings.app.lib.cloudflare.models import VectorPayloadItem
from embeddings.app.lib.cloudflare.models import CreateDatabaseRecord

from embeddings.app.lib.cloudflare.api import CloudflareEmbeddingModels

from embeddings.app.config import settings


def run():
    print(str(CloudflareEmbeddingModels.BAAIBase))
    print(CloudflareEmbeddingModels.BAAIBase.dimensionality)
    exit()
    api = API(
        api_token=settings.CLOUDFLARE_API_TOKEN,
        account_id=settings.CLOUDFLARE_API_ACCOUNT_ID
    )

    # res = api.create_database_table(
    #     settings.CLOUDFLARE_D1_DATABASE_IDENTIFIER,
    #     "test3"
    # )
    # print(("table.create.res", res))

    test_records = [{"source": "abc", "vector_id": "123"},{"source": "def", "vector_id": "456"}]
    res = api.upsert_database_table_records(
        database_id=settings.CLOUDFLARE_D1_DATABASE_IDENTIFIER,
        table_name="test3",
        records=[CreateDatabaseRecord(source=o.get("source"), vector_id=o.get("vector_id")) for o in test_records]
    )
    print(("vector.indexes.list", res))

    vectors = api.vectors_by_ids(
        vector_index_name='test20',
        ids=['dc1b67f7-3287-4241-ac67-6db02a9145bc']
    )
    print(("vectors", vectors))
    exit()

    # res = api.create_vector_index(
    #     name="test4",
    #     preset="@cf/baai/bge-large-en-v1.5"
    # )
    # print(("vector.index.create", res))

    res = api.embed(
        model="@cf/baai/bge-large-en-v1.5",
        texts=["this is a test embedding"]
    )
    print(("embedding.res", res))

    query_vectors = res.get('data', [])
    print(("query_vectors", query_vectors))
    res = api.insert_vectors(
        vector_index_name="test4",
        vectors=[VectorPayloadItem(**{"values": o}) for o in query_vectors]
    )
    print(("vectors.insert.res", res))


if __name__ == "__main__":
    run()
