import uuid

from pydantic import BaseModel
from pydantic import Field

from typing import List, Dict, Any, Optional

from embeddings.models import Pagination


class EmbeddingRead(BaseModel):
    id: str
    vector: Optional[List[float]] = None


class EmbeddingPagination(Pagination):
    items: List[EmbeddingRead]


class EmbeddingsCreate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: List[str]
    payload: Optional[Dict[str, Any]] = None
