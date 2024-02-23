import aiohttp

from typing import List, Dict, Any

from embeddings.app.config import settings
from embeddings.app.lib.cloudflare.api import CloudflareEmbeddingModels

from retry import retry


AI_WORKER_API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"


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
