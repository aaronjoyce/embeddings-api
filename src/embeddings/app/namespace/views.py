from typing import Optional
from fastapi import APIRouter
from fastapi import Request
from fastapi import Response
from fastapi import HTTPException
from fastapi import status

from http import HTTPStatus

from .models import CreateNamespaceResponse
from .models import CreateNamespacePayload
from .models import RetrieveNamespaceData

from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings

router = APIRouter(prefix="/namespace")

client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT)


@router.post("", response_model=CreateNamespaceResponse)
async def create(data_in: CreateNamespacePayload, request: Request, response: Response):
    result = await client.create_collection(
        collection_name=data_in.name,
        vectors_config=VectorParams(size=data_in.dimensionality, distance=data_in.distance),
    )
    print(("result", result))
    return CreateNamespaceResponse(
        success=True
    )


@router.get("/{namespace}", response_model=RetrieveNamespaceData)
async def get(namespace: str, request: Request, response: Response):
    try:
        result = await client.get_collection(
            collection_name=namespace
        )
    except UnexpectedResponse as ex:
        if ex.status_code == HTTPStatus.NOT_FOUND.value:
            response.status_code = HTTPStatus.NOT_FOUND.value
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=[{"msg": f"Collection with name {namespace} not found"}]
            )

        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=[{"msg": ex.content.decode('utf-8')}]
        )

    data = {
        "name": namespace,
        "dimensionality": result.config.params.vectors.size,
        "distance": str(result.config.params.vectors.distance),
        "status": str(result.status),
        "shard_number": result.config.params.shard_number,
        "replication_factor": result.config.params.replication_factor,
        "write_consistency_factor": result.config.params.write_consistency_factor,
        "vectors_count": result.vectors_count,
        "points_count": result.points_count
    }
    print(("retrieve.collection.data", data))
    return RetrieveNamespaceData(
       **data
    )
