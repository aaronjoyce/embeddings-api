from embeddings.app.lib.cloudflare.api import API
from embeddings.app.lib.cloudflare.models import VectorPayloadItem

from embeddings.app.config import settings


def run():
    api = API(
        api_token=settings.CLOUDFLARE_MASTER_API_TOKEN,
        account_id=settings.CLOUDFLARE_API_ACCOUNT_ID
    )
    res = api.list_vector_indexes()
    print(("vector.indexes.list", res))

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
