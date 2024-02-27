import re
import time

from fastapi import status
from fastapi.testclient import TestClient

from embeddings.app.lib.cloudflare.api import CloudflareEmbeddingModels

from .main import app

client = TestClient(app)

CREATE_CLOUDFLARE_NAMESPACE_PATH = "/api/v1/namespace/cloudflare"
CREATE_CLOUDFLARE_EMBEDDING_PATH = "/api/v1/embeddings/cloudflare"
GET_CLOUDFLARE_EMBEDDING_PATH = "/api/v1/embeddings/cloudflare/{namespace}/{embedding_id}"


def generate_namespace_name():
    return f"test_{str(int(time.time()))}"


def test_create_cloudflare_namespace():
    namespace_name = generate_namespace_name()
    response = client.post(
        url=CREATE_CLOUDFLARE_NAMESPACE_PATH,
        json={
            "preset": str(CloudflareEmbeddingModels.BAAISmall),
            "name": namespace_name
        }
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.json().get("name") == namespace_name
    assert response.json().get("dimensionality") == CloudflareEmbeddingModels.BAAISmall.dimensionality


def test_create_cloudflare_namespace_invalid_preset():
    namespace_name = generate_namespace_name()
    response = client.post(
        url=CREATE_CLOUDFLARE_NAMESPACE_PATH,
        json={
            "preset": "invalid_namespace_name",
            "name": namespace_name
        }
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    response_type = response.json().get('detail')[-1].get('type')
    assert response_type == "literal_error"


def test_create_cloudflare_namespace_default_preset():
    namespace_name = generate_namespace_name()
    response = client.post(
        url=CREATE_CLOUDFLARE_NAMESPACE_PATH,
        json={
            "name": namespace_name
        }
    )
    # check dimensionality
    dimensionality = CloudflareEmbeddingModels.BAAIBase.dimensionality
    assert response.json().get("dimensionality") == dimensionality


def test_create_cloudflare_embedding_existing_namespace():
    # first, create a namespace
    namespace_name = generate_namespace_name()

    embedding_model_name = str(CloudflareEmbeddingModels.BAAISmall)
    response = client.post(
        url=CREATE_CLOUDFLARE_NAMESPACE_PATH,
        json={
            "preset": embedding_model_name,
            "name": namespace_name
        }
    )
    assert response.status_code == status.HTTP_200_OK

    response = client.post(
        url=f"{CREATE_CLOUDFLARE_EMBEDDING_PATH}/{namespace_name}",
        json={
            "inputs": [{
                "text": "sample text",
                "persist_source": True
            }],
            "create_namespace": False,
            "embedding_model": embedding_model_name
        }
    )
    assert response.status_code == status.HTTP_200_OK


def test_create_cloudflare_embedding_non_existing_namespace_auto_create():
    namespace_name = generate_namespace_name()
    embedding_model_name = str(CloudflareEmbeddingModels.BAAISmall)

    response = client.post(
        url=f"{CREATE_CLOUDFLARE_EMBEDDING_PATH}/{namespace_name}",
        json={
            "inputs": [{
                "text": "sample text",
                "persist_source": True
            }],
            "create_namespace": True,
            "embedding_model": embedding_model_name
        }
    )
    assert response.status_code == status.HTTP_200_OK


def test_create_cloudflare_embedding_non_existing_namespace_no_create():
    namespace_name = generate_namespace_name()
    embedding_model_name = str(CloudflareEmbeddingModels.BAAISmall)

    response = client.post(
        url=f"{CREATE_CLOUDFLARE_EMBEDDING_PATH}/{namespace_name}",
        json={
            "inputs": [{
                "text": "sample text",
            }],
            "create_namespace": False,
            "embedding_model": embedding_model_name
        }
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND

    response_error_message = response.json().get("detail")[0].get("msg")
    assert bool(re.match(r"vector index with name", response_error_message.lower()))


def test_create_cloudflare_embedding_payload():
    namespace_name = generate_namespace_name()

    embedding_payload = {
        "a": 1,
        "b": 2
    }
    response = client.post(
        url=f"{CREATE_CLOUDFLARE_EMBEDDING_PATH}/{namespace_name}",
        json={
            "inputs": [{
                "text": "sample text",
                "persist_source": True,
                "payload": embedding_payload
            }],
            "create_namespace": True,
        }
    )
    assert response.status_code == status.HTTP_200_OK

    create_response_json = response.json()
    created_embedding_item = create_response_json.get("items", [])[0]

    response = client.get(
        url=GET_CLOUDFLARE_EMBEDDING_PATH.format(
            namespace=namespace_name,
            embedding_id=created_embedding_item.get("id")
        ),
    )
    return response.json() == embedding_payload


def test_create_cloudflare_embedding_source():
    namespace_name = generate_namespace_name()

    source_text = "sample text"
    response = client.post(
        url=f"{CREATE_CLOUDFLARE_EMBEDDING_PATH}/{namespace_name}",
        json={
            "inputs": [{
                "text": source_text,
                "persist_source": True,
            }],
            "create_namespace": True,
        }
    )
    assert response.status_code == status.HTTP_200_OK

    create_response_json = response.json()
    created_embedding_json = create_response_json.get("items", [])[0]

    response = client.get(
        url=GET_CLOUDFLARE_EMBEDDING_PATH.format(
            namespace=namespace_name,
            embedding_id=created_embedding_json.get("id")
        ),
    )
    return response.json().get("source") == source_text
