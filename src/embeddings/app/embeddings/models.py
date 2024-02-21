from pydantic import BaseModel
from pydantic import Field

from typing import List


class BaseEmbeddingsResponse(BaseModel):
    success: bool = Field(default=True)


class GetEmbeddingsResponse(BaseEmbeddingsResponse):
    pass


class CreateEmbeddingsResponse(BaseEmbeddingsResponse):
    pass


class CreateEmbeddingsDataIn(BaseModel):
    text: List[str]
