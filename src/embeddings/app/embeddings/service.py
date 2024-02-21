import requests

from typing import Literal, List

from app.config import settings

CloudflareEmbeddingModel = Literal["@cf/baai/bge-base-en-v1.5"]


API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"


def create_cloudflare_embedding(model: CloudflareEmbeddingModel, text: List[str]):
    headers = {"Authorization": f"Bearer {settings.CLOUDFLARE_API_TOKEN}"}

    res = requests.post(API_BASE_URL.format(
        account_id=settings.CLOUDFLARE_API_ACCOUNT_ID,
        model=model
    ), headers=headers, json={"text": text})
    print(("res", res))
    return res.json()
