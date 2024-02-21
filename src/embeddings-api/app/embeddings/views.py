from fastapi import APIRouter
from fastapi import Request
from fastapi import Response

from pydantic import BaseModel
from pydantic import Field

router = APIRouter(prefix="/embeddings")


class GetEmbeddingsResponse(BaseModel):
    success: bool = Field(default=True)


@router.get("", response_model=GetEmbeddingsResponse)
async def get(request: Request, response: Response):
    return GetEmbeddingsResponse()
