from pydantic import BaseModel
from pydantic import Field


class BaseEmbeddingsResponse(BaseModel):
    success: bool = Field(default=True)


class GetEmbeddingsResponse(BaseEmbeddingsResponse):
    pass


class CreateEmbeddingsResponse(BaseEmbeddingsResponse):
    pass
