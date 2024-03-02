from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel
from pydantic import Field

from app.lib.cloudflare.models import ModelPreset

from app.lib.cloudflare.api import CloudflareEmbeddingModels

from ..models import NamespaceBaseModel

Metric = Literal["cosine", "euclidean", "dot-product"]


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

