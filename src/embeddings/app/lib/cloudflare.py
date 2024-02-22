import requests

from typing import List

from embeddings.app.config import settings
from embeddings.app.config import CloudflareEmbeddingModels

from retry import retry


API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"


@retry(tries=5, delay=1, backoff=1, jitter=0.5)
def embed(model: CloudflareEmbeddingModels, text: List[str]):
    headers = {"Authorization": f"Bearer {settings.CLOUDFLARE_API_TOKEN}"}

    uri = API_BASE_URL.format(
        account_id=settings.CLOUDFLARE_API_ACCOUNT_ID,
        model=model
    )
    res = requests.post(uri, headers=headers, json={"text": text})
    return res.json()
