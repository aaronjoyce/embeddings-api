from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from qdrant_client.http.models import Distance

from qdrant_client.http.models import CollectionStatus


class RetrieveNamespaceBaseModel(BaseModel):
    name: str


class CreateNamespacePayload(RetrieveNamespaceBaseModel):
    dimensionality: int = Field(default=1024)
    distance: Distance = Field(default=Distance.DOT)


class CreateNamespaceResponse(BaseModel):
    success: bool


class RetrieveNamespaceData(RetrieveNamespaceBaseModel):
    dimensionality: int
    distance: Distance
    status: CollectionStatus
    shard_number: int
    replication_factor: int
    write_consistency_factor: int
    vectors_count: int
    points_count: int
