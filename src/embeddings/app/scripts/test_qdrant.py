import uuid

import requests

from typing import List, Optional

NAMESPACE_NAME = "test21"

BASE_URI = "http://localhost:8000"
API_PREFIX = "/api/v1"


def url(path: str, path_id: Optional[str] = None):
    uri = f"{BASE_URI}{API_PREFIX}{path}"
    return f"{uri}/{path_id}" if path_id is not None else uri


def headers():
    return {
        "Authorization": "basic def"
    }


def create_namespace(namespace: str, dimensionality: int = 1024, distance: str = None, ):
    data = {
        "dimensionality": dimensionality,
        "name": namespace,
    }
    if distance is not None:
        data["distance"] = distance

    print(("create.namespace.data", data))
    response = requests.post(
        url=url(path="/namespace/qdrant"),
        json=data
    )
    response_data = response.json()
    print(("response_data", response_data))
    return response_data

def get_embedding(namespace: str, embedding_id: str):
    res = requests.get(
        url=url(path=f"/embeddings/cloudflare/{namespace}", path_id=embedding_id),
        headers=headers()
    )
    print(("get_embedding.res", res))
    return res.json()


def delete_embedding(namespace: str, embedding_id: str):
    res = requests.delete(
        url=url(path=f"/embeddings/cloudflare/{namespace}", path_id=embedding_id),
        headers=headers()
    )
    return res.json()


def create_embedding(namespace: str, text: str):
    json_data = {
        "inputs": [{
            "id": str(uuid.uuid4()),
            "text": text,
            "persist_original": True,
            "payload": {
                "test1": 1
            }
        }],
        "create_namespace": True,
        # "embedding_model": "@cf/baai/bge-base-en-v1.5",  # "@cf/baai/bge-small-en-v1.5"
    }
    print(("json_data", json_data))
    res = requests.post(
        url=url(path="/embeddings/cloudflare", path_id=namespace),
        json=json_data,
        headers=headers()
    )
    return res.json()


def get_namespace(name: str):
    uri = url(path="/namespace/cloudflare", path_id=name)
    res = requests.get(
        url=uri,
        headers=headers()
    )
    return res.json()


def delete_namespace(name: str):
    uri = url(path="/namespace/cloudflare", path_id=name)
    res = requests.delete(
        url=uri,
        headers=headers()
    )
    return res.json()


def query(namespace: str, inputs: str):
    uri = url(path=f"/namespace/cloudflare/{namespace}/query")
    res = requests.post(
        url=uri,
        json={
            "inputs": inputs,
            "return_vectors": False,
            "limit": 1
        },
        headers=headers()
    )
    return res


def run():
    create_namespace(
        namespace="test109",
    )
    insertion_text = ["sample text"]
    res = create_embedding(namespace=NAMESPACE_NAME, text=insertion_text[0])
    print(("cloudflare.embedding.create", res))
    exit()

    res = query(namespace=NAMESPACE_NAME, inputs="abc")
    print(("cloudflare.query.res", res, res.json()))

    exit()

    res = delete_embedding(
        namespace=NAMESPACE_NAME,
        embedding_id="1e24a2fa-e228-4699-a283-fd85142692a6"
    )
    print(("res", res))
    exit()
    res = get_embedding(
        namespace=NAMESPACE_NAME,
        embedding_id="05b1c99e-e947-4e4d-8fc6-5e84947a338d"
    )
    print(("get_embedding.res", res))
    exit()
    # exit()
    create_namespace(
        namespace="invalid_namespace_name"
    )
    exit()


    res = get_namespace(name=NAMESPACE_NAME)
    print(("cloudflare.namespace", res))
    exit()

    res = query(namespace=NAMESPACE_NAME, inputs="sample text")
    print(("cloudflare.query.res", res, res.json()))

    res = delete_namespace(name=NAMESPACE_NAME)
    print(("cloudflare.delete", res))


if __name__ == "__main__":
    run()
