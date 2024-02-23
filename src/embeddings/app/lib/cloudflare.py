import uuid
import json
import aiohttp
import CloudFlare

from pydantic import BaseModel, Field

from typing import List, Dict, Any, Literal, Optional

from embeddings.app.config import settings
from embeddings.app.config import CloudflareEmbeddingModels

from retry import retry


AI_WORKER_API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

ModelPreset = Literal["@cf/baai/bge-small-en-v1.5", "@cf/baai/bge-base-en-v1.5", "@cf/baai/bge-large-en-v1.5"]


class VectorPayloadItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    values: List[float]
    metadata: Dict[str, Any] = {}


def auth_headers() -> Dict[str, Any]:
    return {"Authorization": f"Bearer {settings.CLOUDFLARE_MASTER_API_TOKEN}"}


@retry(tries=5, delay=1, backoff=1, jitter=0.5)
async def aembed(model: CloudflareEmbeddingModels, text: List[str]):
    uri = AI_WORKER_API_BASE_URL.format(
        account_id=settings.CLOUDFLARE_API_ACCOUNT_ID,
        model=model
    )
    async with aiohttp.ClientSession() as session:
        async with session.post(uri, json={
            "text": text
        }, headers=auth_headers()) as response:
            result = await response.json()
            return result


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
