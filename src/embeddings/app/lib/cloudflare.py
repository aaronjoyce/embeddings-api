import aiohttp
import requests

from typing import List, Dict, Any

from embeddings.app.config import settings
from embeddings.app.config import CloudflareEmbeddingModels

from retry import retry


API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"


def auth_headers() -> Dict[str, Any]:
    return {"Authorization": f"Bearer {settings.CLOUDFLARE_API_TOKEN}"}


@retry(tries=5, delay=1, backoff=1, jitter=0.5)
async def aembed(model: CloudflareEmbeddingModels, text: List[str]):
    uri = API_BASE_URL.format(
        account_id=settings.CLOUDFLARE_API_ACCOUNT_ID,
        model=model
    )
    async with aiohttp.ClientSession() as session:
        async with session.post(uri, json={
            "text": text
        }, headers=auth_headers()) as response:
            result = await response.json()
            return result


@retry(tries=5, delay=1, backoff=1, jitter=0.5)
def embed(model: CloudflareEmbeddingModels, text: List[str]):
    uri = API_BASE_URL.format(
        account_id=settings.CLOUDFLARE_API_ACCOUNT_ID,
        model=model
    )
    res = requests.post(uri, headers=auth_headers(), json={"text": text})
    return res.json()

