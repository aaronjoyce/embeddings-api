from pydantic import BaseModel
from pydantic import Field

from typing import List, Dict, Any, Optional


class Pagination(BaseModel):
    itemsPerPage: Optional[int] = None
    total: int
    page: int


class BaseEmbeddingsResponse(BaseModel):
    success: bool = Field(default=True)


class EmbeddingRead(BaseModel):
    id: str
    vector: Optional[List[float]] = None


class EmbeddingPagination(Pagination):
    items: List[EmbeddingRead]


class GetEmbeddingsResponse(BaseEmbeddingsResponse):
    pass


class CreateEmbeddingsResponse(BaseEmbeddingsResponse):
    pass


class EmbeddingsCreate(BaseModel):
    text: List[str]
    payload: Optional[Dict[str, Any]] = None
