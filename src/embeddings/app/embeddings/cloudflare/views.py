import uuid
import CloudFlare

from fastapi import status
from fastapi import APIRouter
from fastapi import Request
from fastapi import Response
from fastapi import HTTPException

from ..models import EmbeddingRead
from ..models import EmbeddingsCreate

from embeddings.models import InsertionResult
from embeddings.app.config import settings

from embeddings.app.lib.cloudflare.api import ERROR_CODE_VECTOR_INDEX_NOT_FOUND
from embeddings.app.lib.cloudflare.api import API
from embeddings.app.lib.cloudflare.api import CloudflareEmbeddingModels
from embeddings.app.lib.cloudflare.models import VectorPayloadItem

router = APIRouter(prefix="/embeddings/cloudflare")


@router.post("/{namespace}", response_model=InsertionResult[EmbeddingRead])
async def create(namespace: str, data_in: EmbeddingsCreate, request: Request, response: Response):
    cloudflare = API(
        api_token=settings.CLOUDFLARE_MASTER_API_TOKEN,
        account_id=settings.CLOUDFLARE_API_ACCOUNT_ID
    )
    result = cloudflare.embed(
        model=CloudflareEmbeddingModels.BAAIBase.value,
        texts=data_in.text
    )
    query_vectors = result.get('data', [])

    if data_in.persist_decoded:
        vectors = [VectorPayloadItem(**{
            "values": o[0],
            "metadata": {"source": o[1], **data_in.payload}
        }) for o in zip(query_vectors, data_in.text)]
    else:
        vectors = [VectorPayloadItem(**{"values": o, "metadata": data_in.payload}) for o in query_vectors]

    try:
        result = cloudflare.insert_vectors(
            vector_index_name=namespace,
            vectors=vectors,
            create_on_not_found=data_in.create_index
        )
        print(("result", result))
    except CloudFlare.exceptions.CloudFlareAPIError as ex:
        if int(ex) == ERROR_CODE_VECTOR_INDEX_NOT_FOUND:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=[{
                    "msg": f"Vector index with name '{namespace}' not found. "
                    f"Create the index via a separate call or include 'create_index' "
                    f"in your payload to automagically create and insert"
                }]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=[{"msg": str(ex)}]
            )

    items = [EmbeddingRead(id=o) for o in result.get('ids', [])]
    return InsertionResult[EmbeddingRead](
        count=len(items),
        items=items
    )
