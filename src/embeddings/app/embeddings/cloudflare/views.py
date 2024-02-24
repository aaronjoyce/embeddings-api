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
from embeddings.app.lib.cloudflare.models import CreateDatabaseRecord

router = APIRouter(prefix="/embeddings/cloudflare")


@router.post("/{namespace}", response_model=InsertionResult[EmbeddingRead])
async def create(namespace: str, data_in: EmbeddingsCreate, request: Request, response: Response):
    cloudflare = API(
        api_token=settings.CLOUDFLARE_API_TOKEN,
        account_id=settings.CLOUDFLARE_API_ACCOUNT_ID
    )
    result = cloudflare.embed(
        model=CloudflareEmbeddingModels.BAAIBase.value,
        texts=data_in.text
    )
    query_vectors = result.get('data', [])
    vectors = [VectorPayloadItem(**{"values": o, "metadata": data_in.payload}) for o in query_vectors]

    try:
        result = cloudflare.insert_vectors(
            vector_index_name=namespace,
            vectors=vectors,
            create_on_not_found=data_in.create_index
        )
        insertion_records = [CreateDatabaseRecord(vector_id=o[0], source=o[1]) for o in zip(
            result.get('ids', []), data_in.text
        )]
        insertion_result = cloudflare.upsert_database_table_records(
            database_id=settings.CLOUDFLARE_D1_DATABASE_IDENTIFIER,
            table_name=namespace,
            records=insertion_records
        )
        print(("insertion_result", insertion_result))
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
