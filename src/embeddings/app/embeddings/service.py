import requests

from typing import Literal, List

from app.config import settings
from app.config import CloudflareEmbeddingModels


API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"


def create_cloudflare_embedding(model: CloudflareEmbeddingModels, text: List[str]):
    headers = {"Authorization": f"Bearer {settings.CLOUDFLARE_API_TOKEN}"}

    uri = API_BASE_URL.format(
        account_id=settings.CLOUDFLARE_API_ACCOUNT_ID,
        model=model
    )
    print(("uri.2", uri, "headers", headers))

    res = requests.post(uri, headers=headers, json={"text": text})
    print(("res.0", res))
    return res.json()
