from typing import Optional, Generic, TypeVar, List
from pydantic import BaseModel


ItemType = TypeVar('ItemType')


class Pagination(BaseModel):
    itemsPerPage: Optional[int] = None
    total: int
    page: int


class InsertionResult(BaseModel, Generic[ItemType]):
    count: int
    items: List[ItemType]
