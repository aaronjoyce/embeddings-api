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
    shard_number: Optional[int]
    metric: Optional[Metric]
    replication_factor: Optional[int]
    write_consistency_factor: Optional[int]
    vectors_count: Optional[int]
    points_count: Optional[int]


class NamespaceQuery(BaseModel):
    inputs: str
