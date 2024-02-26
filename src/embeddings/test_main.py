import time

from fastapi import status
from fastapi.testclient import TestClient

from embeddings.app.lib.cloudflare.api import CloudflareEmbeddingModels

from .main import app

client = TestClient(app)

CREATE_CLOUDFLARE_NAMESPACE_PATH = "/api/v1/namespace/cloudflare"


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
    assert response.json().get("name") == namespace_name


def test_create_cloudflare_namespace_default_preset():
    namespace_name = generate_namespace_name()
    response = client.post(
        url=CREATE_CLOUDFLARE_NAMESPACE_PATH,
        json={
            "name": namespace_name
        }
    )
    # check dimensionality
    assert response.json().get("dimensionality") == CloudflareEmbeddingModels.BAAIBase.dimensionality
