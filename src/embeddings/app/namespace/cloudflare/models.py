from typing import Optional, Literal
from pydantic import BaseModel
from pydantic import Field

from embeddings.app.lib.cloudflare.models import ModelPreset

from embeddings.app.lib.cloudflare.api import CloudflareEmbeddingModels

Metric = Literal["cosine", "euclidean", "dot-product"]


class NamespaceBaseModel(BaseModel):
    name: str


class NamespaceCreate(NamespaceBaseModel):
    preset: ModelPreset = Field(default=CloudflareEmbeddingModels.BAAIBase.value)


class NamespaceDelete(BaseModel):
    success: bool


class NamespaceRead(NamespaceBaseModel):
    dimensionality: int
    metric: Metric
    shard_number: Optional[int] = None
    replication_factor: Optional[int] = None
    write_consistency_factor: Optional[int] = None
    vectors_count: Optional[int] = None
    points_count: Optional[int] = None


class NamespaceQuery(BaseModel):
    inputs: str
