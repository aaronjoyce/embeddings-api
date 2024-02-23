import uuid
from typing import List, Any, Dict, Literal

from pydantic import BaseModel
from pydantic import Field


ModelPreset = Literal["@cf/baai/bge-small-en-v1.5", "@cf/baai/bge-base-en-v1.5", "@cf/baai/bge-large-en-v1.5"]


class VectorPayloadItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    values: List[float]
    metadata: Dict[str, Any] = {}
