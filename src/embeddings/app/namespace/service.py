from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.async_qdrant_client import AsyncQdrantClient

from embeddings.app.config import settings

from .models import NamespaceRead

client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT)


async def namespace(name: str):
    result = await client.get_collection(
        collection_name=name
    )
    data = {
        "name": name,
        "dimensionality": result.config.params.vectors.size,
        "distance": str(result.config.params.vectors.distance),
        "status": str(result.status),
        "shard_number": result.config.params.shard_number,
        "replication_factor": result.config.params.replication_factor,
        "write_consistency_factor": result.config.params.write_consistency_factor,
        "vectors_count": result.vectors_count,
        "points_count": result.points_count
    }
    return NamespaceRead(
        **data
    )
