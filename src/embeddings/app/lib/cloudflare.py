import uuid
import json
import aiohttp
import requests

from pydantic import BaseModel, Field

from typing import List, Dict, Any, Literal

from embeddings.app.config import settings
from embeddings.app.config import CloudflareEmbeddingModels

from retry import retry


AI_WORKER_API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

ModelPreset = Literal["@cf/baai/bge-small-en-v1.5", "@cf/baai/bge-base-en-v1.5", "@cf/baai/bge-large-en-v1.5"]


class VectorPayloadItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    values: List[float]
    metadata: Dict[str, Any] = {}


def ai_worker_auth_headers() -> Dict[str, Any]:
    return {"Authorization": f"Bearer {settings.CLOUDFLARE_API_TOKEN}"}


def vectorize_auth_headers() -> Dict[str, Any]:
    return {"Authorization": f"Bearer {settings.CLOUDFLARE_API_VECTORIZE_TOKEN}"}


@retry(tries=5, delay=1, backoff=1, jitter=0.5)
async def aembed(model: CloudflareEmbeddingModels, text: List[str]):
    uri = AI_WORKER_API_BASE_URL.format(
        account_id=settings.CLOUDFLARE_API_ACCOUNT_ID,
        model=model
    )
    async with aiohttp.ClientSession() as session:
        async with session.post(uri, json={
            "text": text
        }, headers=ai_worker_auth_headers()) as response:
            result = await response.json()
            return result


@retry(tries=5, delay=1, backoff=1, jitter=0.5)
def embed(model: CloudflareEmbeddingModels, text: List[str]):
    uri = AI_WORKER_API_BASE_URL.format(
        account_id=settings.CLOUDFLARE_API_ACCOUNT_ID,
        model=model
    )
    res = requests.post(uri, headers=ai_worker_auth_headers(), json={"text": text})
    return res.json()


def create_vector_index(name: str, model_preset: ModelPreset):
    url = "https://api.cloudflare.com/client/v4/accounts/{account_id}/vectorize/indexes".format(
        account_id=settings.CLOUDFLARE_API_ACCOUNT_ID
    )
    headers = {
        **vectorize_auth_headers()
    }
    result = requests.post(
        url=url,
        headers=headers,
        json={
            "config": {
                "preset": model_preset
            },
            "name": name
        }
    )
    return result


def insert_vectors(vector_index_name: str, vectors: List[VectorPayloadItem]):
    data = "\n".join([json.dumps({"id": o.id, "values": o.values, "metadata": o.metadata}) for o in vectors])
    url = "https://api.cloudflare.com/client/v4/accounts/{account_id}/vectorize/indexes/{index_name}/insert".format(
        account_id=settings.CLOUDFLARE_API_ACCOUNT_ID,
        index_name=vector_index_name
    )
    headers = {
        "Content-Type": "application/x-ndjson",
        **vectorize_auth_headers()
    }
    result = requests.post(url=url, headers=headers, data=data)
    return result


def list_vector_indexes():
    url = "https://api.cloudflare.com/client/v4/accounts/{account_identifier}/vectorize/indexes".format(
        account_identifier=settings.CLOUDFLARE_API_ACCOUNT_ID
    )
    headers = vectorize_auth_headers()
    result = requests.get(
        url=url,
        headers=headers
    )
    return result
