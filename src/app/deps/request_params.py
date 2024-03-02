from typing import Annotated

from fastapi import Depends
from fastapi import Query


def common_params(
    page: int = Query(default=1, gte=1, lt=2147483647),
    limit: int = Query(default=10, gte=1, lt=2147483647)
):
    offset = (page - 1) * limit
    return {
        "page": page,
        "limit": limit,
        "offset": offset
    }


CommonParams = Annotated[dict[int], Depends(common_params)]