from typing import Optional
from pydantic import BaseModel


class Pagination(BaseModel):
    itemsPerPage: Optional[int] = None
    total: int
    page: int
