import requests

from typing import Optional, List


NAMESPACE_NAME = "test20"

BASE_URI = "http://localhost:8000"
API_PREFIX = "/api/v1"


def url(path: str, path_id: Optional[str] = None):
    uri = f"{BASE_URI}{API_PREFIX}{path}"
    return f"{uri}/{path_id}" if path_id is not None else uri


def delete_namespace(name: str):
    res = requests.delete(
        url=url(path="/namespace", path_id=name)
    )
    return res


def create_namespace(name: str, dimensionality: int = 768):
    res = requests.post(
        url=url(path="/namespace"),
        json={
            "name": name,
            "dimensionality": dimensionality
        }
    )
    return res.json()


def qdrant_create_embedding(namespace: str, text: List[str]):
    res = requests.post(
        url=url(path="/embeddings/qdrant", path_id=namespace),
        json={
            "text": text
        }
    )
    return res.json()


def cloudflare_create_embedding(namespace: str, text: List[str]):
    res = requests.post(
        url=url(path="/embeddings/cloudflare", path_id=namespace),
        json={
            "text": text,
            "create_index": True,
            "persist_decoded": True,
            "payload": {
                "test1": 1
            }
        }
    )
    return res.json()


def qdrant_list_embeddings(namespace: str, page: int = None, limit: int = None):
    args = {
        "url": url(path="/embeddings/qdrant", path_id=namespace),
    }
    params = {}
    if page is not None:
        params["page"] = page

    print(("embeddings.limit", limit))
    if limit is not None:
        params["limit"] = limit

    if params:
        args["params"] = params

    print(("list.args", args))
    res = requests.get(**args)
    return res.json()


def qdrant_get_embedding(namespace: str, embedding_id: str):
    uri = url(path=f"/embeddings/qdrant/{namespace}", path_id=embedding_id)
    print(("get_embedding.uri", uri, ))
    res = requests.get(
        url=uri
    )
    return res.json()


def query(namespace: str, inputs: str):
    uri = url(path=f"/namespace/{namespace}/query")
    print(("uri", uri))
    res = requests.post(
        url=uri,
        json={
            "inputs": inputs
        }
    )
    print(("res", res, ))
    return res.json()


def run():
    insertion_text = ["this is some sample text"]
    res = cloudflare_create_embedding(namespace="test20", text=insertion_text)
    print(("cloudflare.embedding.create", res))

    res = delete_namespace(name=NAMESPACE_NAME)
    print(("qdrant.namespace.delete", res))

    res = create_namespace(name=NAMESPACE_NAME)
    print(("namespace.create.res", res))

    for i in range(3):
        res = qdrant_create_embedding(namespace=NAMESPACE_NAME, text=insertion_text)
        print((f"embedding.create.res-{i}", res))

    res = qdrant_list_embeddings(namespace=NAMESPACE_NAME, limit=1, page=1)
    print(("embeddings.list.res", res))

    embedding_id = res.get('items', [])[0].get('id')
    print(("embedding_id", embedding_id, ))
    res = qdrant_get_embedding(namespace=NAMESPACE_NAME, embedding_id=embedding_id)
    print(("embedding.get.res", res))

    res = query(namespace=NAMESPACE_NAME, inputs="some sample text")
    print(("namespace.query.res", res, ))


if __name__ == "__main__":
    run()
