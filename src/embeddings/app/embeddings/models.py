import uuid

from pydantic import BaseModel, field_validator, ValidationInfo
from pydantic import Field

from typing import List, Dict, Any, Optional

from embeddings.models import Pagination

from embeddings.app.lib.cloudflare.api import CloudflareEmbeddingModels
from embeddings.app.lib.cloudflare.api import MAX_EMBEDDING_INPUT_TOKENS


class EmbeddingDelete(BaseModel):
    success: bool
    count: Optional[int] = None


class EmbeddingRead(BaseModel):
    id: str
    vector: Optional[List[float]] = None
    payload: Optional[Dict[str, Any]] = None
    source: Optional[str] = None


class EmbeddingPagination(Pagination):
    items: List[EmbeddingRead]


class EmbeddingsCreateSingle(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(max)
    payload: Optional[Dict[str, Any]] = Field(default_factory=dict)
    persist_original: Optional[bool] = Field(default=False)


class EmbeddingCreateMulti(BaseModel):
    create_namespace: Optional[bool] = Field(default=True)
    embedding_model: Optional[CloudflareEmbeddingModels] = Field(default=CloudflareEmbeddingModels.BAAIBase)
    inputs: List[EmbeddingsCreateSingle]

    @field_validator('inputs')
    @classmethod
    def check_text_length(cls, v: List[EmbeddingsCreateSingle]) -> List[EmbeddingsCreateSingle]:
        for o in v:
            token_count = len(o.text.split(' '))
            if token_count > MAX_EMBEDDING_INPUT_TOKENS:
                raise ValueError(
                    f"Text token count ({token_count}) exceeds the maximum supported "
                    f"by the embedding model: {MAX_EMBEDDING_INPUT_TOKENS}."
                )
        return v
