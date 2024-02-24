from enum import Enum

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "embeddings"

    PROJECT_DESCRIPTION: Optional[str] = None

    API_PATH: str = "/api/v1"

    # Cloudflare
    CLOUDFLARE_API_ACCOUNT_ID: str
    CLOUDFLARE_API_TOKEN: str
    CLOUDFLARE_D1_DATABASE_IDENTIFIER: str

    # Qdrant
    QDRANT_HOST: str
    QDRANT_HTTP_PORT: int
    QDRANT_GRPC_PORT: int


settings = Settings()
