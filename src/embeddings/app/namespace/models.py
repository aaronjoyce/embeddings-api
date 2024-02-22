from pydantic import BaseModel
from pydantic import Field


class CreateNamespacePayload(BaseModel):
    name: str
    dimensions: int = Field(default=1024)


class CreateNamespaceResponse(BaseModel):
    success: bool


class RetrieveNamespaceResponse(BaseModel):
    name: str
