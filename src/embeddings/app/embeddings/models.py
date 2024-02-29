import uuid

from pydantic import BaseModel, field_validator, ValidationInfo
from pydantic import Field

from typing import List, Dict, Any, Optional

from embeddings.models import Pagination

from embeddings.app.lib.cloudflare.api import CloudflareEmbeddingModels


class EmbeddingDelete(BaseModel):
    success: bool


class EmbeddingRead(BaseModel):
    id: str
    vector: Optional[List[float]] = None
    payload: Optional[Dict[str, Any]] = None
    source: Optional[str] = None


class EmbeddingPagination(Pagination):
    items: List[EmbeddingRead]


class EmbeddingsCreateSingle(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    payload: Optional[Dict[str, Any]] = Field(default_factory=dict)
    persist_original: Optional[bool] = Field(default=False)


class EmbeddingCreateMulti(BaseModel):
    inputs: List[EmbeddingsCreateSingle]
    create_namespace: Optional[bool] = Field(default=True)
    embedding_model: Optional[CloudflareEmbeddingModels] = Field(default=CloudflareEmbeddingModels.BAAIBase)
