import json
import enum
import CloudFlare

from typing import Optional, List

from retry import retry

from embeddings.app.config import settings

from .models import VectorPayloadItem


# Error codes
ERROR_CODE_VECTOR_INDEX_NOT_FOUND = 3000
ERROR_CODE_INSERT_VECTOR_INDEX_SIZE_MISMATCH = 4003

class CloudflareEmbeddingModels(enum.Enum):
    BAAISmall = "@cf/baai/bge-small-en-v1.5"
    BAAIBase = "@cf/baai/bge-base-en-v1.5"
    BAAILarge = "@cf/baai/bge-large-en-v1.5"


SMALL_DIMENSION = 384
BASE_DIMENSION = 768
LARGE_DIMENSION = 1024

DIMENSIONALITY_PRESETS = {
    SMALL_DIMENSION: [CloudflareEmbeddingModels.BAAISmall],
    BASE_DIMENSION: [CloudflareEmbeddingModels.BAAIBase],
    LARGE_DIMENSION: [CloudflareEmbeddingModels.BAAILarge]
}

class API:

    def __init__(self, api_token: str, account_id: str):
        self.api_token = api_token
        self.account_id = account_id
        self.client = CloudFlare.CloudFlare(token=self.api_token)

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def create_vector_index(self, name: str, preset: str, description: Optional[str] = None):
        data = {
            "name": name,
            "config": {
                "preset": preset
            }
        }
        if description is not None:
            data["description"] = description

        res = self.client.accounts.vectorize.indexes.post(
            self.account_id,
            data=data
        )
        return res

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def list_vector_indexes(self):
        res = self.client.accounts.vectorize.indexes(
            self.account_id
        )
        return res

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def vectors_by_ids(self, vector_index_name: str, ids: List[str]):
        res = self.client.accounts.vectorize.indexes.get_by_ids.post(
            settings.CLOUDFLARE_API_ACCOUNT_ID,
            vector_index_name,
            data={
                "ids": ids
            }
        )
        return res

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def insert_vectors(self, vector_index_name: str, vectors: List[VectorPayloadItem], create_on_not_found: bool = False):
        data = "\n".join([json.dumps({"id": o.id, "values": o.values, "metadata": o.metadata}) for o in vectors])
        try:
            res = self.client.accounts.vectorize.indexes.insert.post(
                settings.CLOUDFLARE_API_ACCOUNT_ID,
                vector_index_name,
                data=data
            )
        except CloudFlare.exceptions.CloudFlareAPIError as ex:
            if int(ex) == ERROR_CODE_VECTOR_INDEX_NOT_FOUND and create_on_not_found:
                # infer dimensionality from the vector at index 0
                default_dimensionality_presets = DIMENSIONALITY_PRESETS.get(len(vectors[0].values), [])
                if not default_dimensionality_presets:
                    allowed_dimensionality_values = ','.join([str(o) for o in DIMENSIONALITY_PRESETS.keys()])
                    raise Exception(
                        f"Unsupported vector preset dimensionality. "
                        f"Expected one of: {','.join(allowed_dimensionality_values)}, got: {len(vectors[0].values)}"
                    )

                create_vector_index_res = self.create_vector_index(
                    name=vector_index_name,
                    preset=default_dimensionality_presets[0].value
                )
                return self.insert_vectors(
                    vector_index_name=vector_index_name,
                    vectors=vectors
                )

            raise ex

        return res

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def embed(self, model, texts: List[str]):
        res = self.client.accounts.ai.run.post(
            settings.CLOUDFLARE_API_ACCOUNT_ID,
            model,
            data={
                "text": texts
            }
        )
        return res
