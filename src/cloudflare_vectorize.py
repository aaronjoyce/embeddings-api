import json
import uuid

from embeddings.app.lib.cloudflare import embed

from embeddings.app.config import CloudflareEmbeddingModels

from embeddings.app.config import settings

from embeddings.app.lib.cloudflare import list_vector_indexes
from embeddings.app.lib.cloudflare import insert_vectors
from embeddings.app.lib.cloudflare import create_vector_index
from embeddings.app.lib.cloudflare import VectorPayloadItem

import CloudFlare

from embeddings.app.config import settings


def run():
    cf = CloudFlare.CloudFlare(
        token=settings.CLOUDFLARE_MASTER_API_TOKEN
    )
    res = cf.accounts.vectorize.indexes(
        settings.CLOUDFLARE_API_ACCOUNT_ID
    )
    print(("res", res))

    embedded_query_result = embed(model=CloudflareEmbeddingModels.BAAIBase.value, text=["this is some example text"])
    print(("embedded_query_result", embedded_query_result))
    query_vectors = embedded_query_result.get('result', {}).get('data', [])
    print(("query_vectors", query_vectors))

    # insert_vectors(vector_index_name="test1", vectors=[VectorPayloadItem(**{"values": query_vector})])
    data = "\n".join([json.dumps({"id": str(uuid.uuid4()), "values": o}) for o in query_vectors])

    vector_index_name = "test1"
    res2 = cf.accounts.vectorize.indexes.insert.post(
        settings.CLOUDFLARE_API_ACCOUNT_ID,
        vector_index_name,
        data=data
    )
    print(("res2", res2))

    list_vector_indexes()
    # create_vector_index(name="test1", model_preset="@cf/baai/bge-base-en-v1.5")


if __name__ == "__main__":
    run()
