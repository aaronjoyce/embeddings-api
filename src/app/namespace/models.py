from typing import Optional, Dict, Any, List

from pydantic import BaseModel
from pydantic import Field

from qdrant_client.http.models import Distance
from qdrant_client.http.models import CollectionStatus

from app.models import Pagination


class NamespaceBaseModel(BaseModel):
    name: str


class NamespaceDelete(BaseModel):
    success: bool


class NamespaceRead(NamespaceBaseModel):
    dimensionality: int
    distance: Distance
    status: CollectionStatus
    shard_number: int
    replication_factor: int
    write_consistency_factor: int
    vectors_count: int
    points_count: int


class NamespaceQuery(BaseModel):
    inputs: str
    return_vectors: Optional[bool] = False
    return_metadata: Optional[bool] = False
    limit: Optional[int] = Field(default=5, gt=0)
    filter: Optional[Dict[str, Any]] = Field(default=None)


class NamespacePagination(Pagination):
    items: List[NamespaceBaseModel]
