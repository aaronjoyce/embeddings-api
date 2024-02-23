from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from qdrant_client.http.models import VectorStruct

from embeddings.models import Pagination


class DocumentRead(BaseModel):
    id: str
    payload: Optional[Dict[str, Any]]
    score: Optional[float]
    vector: Optional[VectorStruct]


class DocumentPagination(Pagination):
    items: List[DocumentRead]