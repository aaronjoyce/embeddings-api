from typing import Annotated
from fastapi import Depends

from app.config import settings

from app.lib.cloudflare.api import API


def cloudflare_api_client():
    return API(
        api_token=settings.CLOUDFLARE_API_TOKEN,
        account_id=settings.CLOUDFLARE_API_ACCOUNT_ID
    )


CloudflareClient = Annotated[API, Depends(cloudflare_api_client)]

