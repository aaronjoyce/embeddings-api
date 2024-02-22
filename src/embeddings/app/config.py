from enum import Enum

from typing import Optional

from pydantic_settings import BaseSettings


class CloudflareEmbeddingModels(Enum):
    BAAISmall = "@cf/baai/bge-small-en-v1.5"
    BAAIBase = "@cf/baai/bge-base-en-v1.5"
    BAAILarge = "@cf/baai/bge-large-en-v1.5"


class Settings(BaseSettings):
    PROJECT_NAME: str = "embeddings"

    PROJECT_DESCRIPTION: Optional[str] = None

    API_PATH: str = "/api/v1"

    # Cloudflare
    CLOUDFLARE_API_ACCOUNT_ID: str
    CLOUDFLARE_API_TOKEN: str

    # Qdrant
    QDRANT_HOST: str
    QDRANT_HTTP_PORT: int
    QDRANT_GRPC_PORT: int


settings = Settings()
