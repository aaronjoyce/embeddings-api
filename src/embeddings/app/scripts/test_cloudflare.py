import uuid

import requests

from typing import List, Optional

NAMESPACE_NAME = "test21"

BASE_URI = "http://localhost:8000"
API_PREFIX = "/api/v1"


def url(path: str, path_id: Optional[str] = None):
    uri = f"{BASE_URI}{API_PREFIX}{path}"
    return f"{uri}/{path_id}" if path_id is not None else uri


def create_embedding(namespace: str, text: str):
    json_data = {
        "inputs": [{
            "id": str(uuid.uuid4()),
            "text": text,
            "persist_source": True,
            "payload": {
                "test1": 1
            }
        }],
        "create_namespace": False,
        # "embedding_model": "@cf/baai/bge-base-en-v1.5",  # "@cf/baai/bge-small-en-v1.5"
    }
    print(("json_data", json_data))
    res = requests.post(
        url=url(path="/embeddings/cloudflare", path_id=namespace),
        json=json_data
    )
    return res.json()


def get_namespace(name: str):
    uri = url(path="/namespace/cloudflare", path_id=name)
    res = requests.get(
        url=uri
    )
    return res.json()


def delete_namespace(name: str):
    uri = url(path="/namespace/cloudflare", path_id=name)
    res = requests.delete(
        url=uri
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
        }
    )
    return res


def run():
    # insertion_text = ["this is some sample text"]
    # res = create_embedding(namespace=NAMESPACE_NAME, text=insertion_text[0])
    # print(("cloudflare.embedding.create", res))
    # exit()

    res = get_namespace(name=NAMESPACE_NAME)
    print(("cloudflare.namespace", res))
    exit()

    res = query(namespace=NAMESPACE_NAME, inputs="sample text")
    print(("cloudflare.query.res", res, res.json()))

    res = delete_namespace(name=NAMESPACE_NAME)
    print(("cloudflare.delete", res))


if __name__ == "__main__":
    run()
