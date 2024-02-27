from typing import Literal, List, Optional, Dict, Any

from pydantic import BaseModel, Field


# need to be handled recursively


class Column(BaseModel):
    comparator: Literal["_eq", "_neq"]
    value: str
    name: str


class Operator(BaseModel):
    value: Literal["_and", "_or", "_not"]
    columns: List[Column]
    operator: Optional["Operator"] = Field(default=None)


class NamespacePermission(BaseModel):
    namespace: str
    role: str
    operator: Operator


def evaluate(permission: NamespacePermission, data: List[Dict[str, Any]], headers: Dict[str, str]):
    pass


def run():
    roles = {
        "namespace1": {
            "operator": {
                "value": "and" or "or",
                "columns": [{
                    "comparator": "_eq" or "_neq",
                    "value": "X-Hasura-User-Id"
                }],
                "operator": ...
            },
        }
    }

    namespace_permission = NamespacePermission(
        namespace="test1",
        role="admin",
        operator=Operator(
            value="_and",
            columns=[
                Column(
                    comparator="_eq",
                    name="userId",
                    value="X-Synapse-User-Id"
                ),
                Column(
                    comparator="_neq",
                    name="teamId",
                    value="X-Synapse-Team-Id"
                )
            ],
            operator=Operator(
                value="_and",
                columns=[
                    Column(
                        comparator="_eq",
                        name="userId",
                        value="X-Synapse-User-Id"
                    ),
                    Column(
                        comparator="_neq",
                        name="teamId",
                        value="X-Synapse-Team-Id"
                    )
                ]

            )
        )
    )
    print(("namespace_permission", namespace_permission.dict(), ))

    headers = {
        'X-Synapse-Team-Id': 'abc',
        'X-Synapse-User-Id': '123',
    }
    test_data = [
        {
            "userId": '123',
            "teamId": 'abc'
        },
        {
            "userId": '123',
            "teamId": 'def'
        },
        {
            "userId": '123',
            "teamId": 'ghi'
        },
        {
            "userId": '456',
            "teamId": 'jkl'
        }
    ]
    evaluate(
        namespace_permission,
        test_data,
        headers
    )


if __name__ == "__main__":
    run()
