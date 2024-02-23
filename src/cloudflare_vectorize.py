from embeddings.app.lib.cloudflare import embed

from embeddings.app.config import CloudflareEmbeddingModels

from embeddings.app.config import settings

from embeddings.app.lib.cloudflare import list_vector_indexes
from embeddings.app.lib.cloudflare import insert_vectors
from embeddings.app.lib.cloudflare import create_vector_index
from embeddings.app.lib.cloudflare import VectorPayloadItem


def run():
    list_vector_indexes()
    # create_vector_index(name="test1", model_preset="@cf/baai/bge-base-en-v1.5")

    embedded_query_result = embed(model=CloudflareEmbeddingModels.BAAIBase.value, text=["this is some example text"])
    query_vectors = embedded_query_result.get('result', {}).get('data', [])
    query_vector = query_vectors[0]
    insert_vectors(vector_index_name="test1", vectors=[VectorPayloadItem(**{"values": query_vector})])


if __name__ == "__main__":
    run()
