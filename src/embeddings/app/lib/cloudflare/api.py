import json
import CloudFlare

from typing import Optional, List

from retry import retry

from embeddings.app.config import settings

from .models import VectorPayloadItem


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
    def insert_vectors(self, vector_index_name: str, vectors: List[VectorPayloadItem]):
        data = "\n".join([json.dumps({"id": o.id, "values": o.values, "metadata": o.metadata}) for o in vectors])
        res = self.client.accounts.vectorize.indexes.insert.post(
            settings.CLOUDFLARE_API_ACCOUNT_ID,
            vector_index_name,
            data=data
        )
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
