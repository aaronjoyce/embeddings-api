import re
import time

from typing import Dict, Any, List
from fastapi import status
from fastapi.testclient import TestClient

from embeddings.app.lib.cloudflare.api import CloudflareEmbeddingModels

from .main import app

client = TestClient(app)

CREATE_CLOUDFLARE_NAMESPACE_PATH = "/api/v1/namespace/cloudflare"
CREATE_CLOUDFLARE_EMBEDDING_PATH = "/api/v1/embeddings/cloudflare"
CLOUDFLARE_EMBEDDING_PATH = "/api/v1/embeddings/cloudflare/{namespace}/{embedding_id}"
DELETE_CLOUDFLARE_NAMESPACE_PATH = "/api/v1/namespace/cloudflare/{namespace}"


EMBEDDING_TEXT = "sample text"


class NamespaceClient:

    namespace: str

    def create(self, name: str, preset: str = None):
        data = {
            "name": name
        }
        if preset is not None:
            data["preset"] = preset

        response = client.post(
            url=CREATE_CLOUDFLARE_NAMESPACE_PATH,
            json=data
        )
        self.namespace = name
        return response

    def delete(self):
        response = client.delete(
            url=DELETE_CLOUDFLARE_NAMESPACE_PATH.format(
                namespace=self.namespace
            )
        )
        return response


def create_cloudflare_namespace(name: str, preset: str = None):
    data = {
        "name": name
    }
    if preset is not None:
        data["preset"] = preset

    response = client.post(
        url=CREATE_CLOUDFLARE_NAMESPACE_PATH,
        json=data
    )
    return response


def create_cloudflare_embedding(
        namespace: str,
        text: str,
        persist_source: bool = True,
        create_namespace: bool = True,
        embedding_model: str = None,
        payload: Dict[str, Any] = None,
):
    data = {
        "inputs": [{
            "text": text,
            "persist_source": persist_source,
            "payload": payload if payload is not None else {}
        }],
        "create_namespace": create_namespace,
    }
    if embedding_model is not None:
        data["embedding_model"] = embedding_model

    response = client.post(
        url=f"{CREATE_CLOUDFLARE_EMBEDDING_PATH}/{namespace}",
        json=data
    )
    return response


def generate_namespace_name():
    return f"test_{str(int(time.time()))}"


class TestBase:

    namespaces: List[str] = []

    def setup_method(self):
        self.namespaces = []

    def teardown_method(self):
        for namespace in self.namespaces:
            client.delete(
                url=DELETE_CLOUDFLARE_NAMESPACE_PATH.format(
                    namespace=namespace
                )
            )


class TestCloudflareEmbedding(TestBase):

    def test_create_existing_namespace(self):
        # first, create a namespace
        namespace_name = generate_namespace_name()

        embedding_model = str(CloudflareEmbeddingModels.BAAISmall)
        response = create_cloudflare_namespace(
            name=namespace_name,
            preset=embedding_model
        )
        assert response.status_code == status.HTTP_200_OK

        response = create_cloudflare_embedding(
            namespace=namespace_name,
            text=EMBEDDING_TEXT,
            persist_source=True,
            create_namespace=False,
            embedding_model=embedding_model
        )
        assert response.status_code == status.HTTP_200_OK
        self.namespaces.append(namespace_name)

    def test_create_non_existing_namespace_auto_create(self):
        namespace_name = generate_namespace_name()
        embedding_model = str(CloudflareEmbeddingModels.BAAISmall)
        response = create_cloudflare_embedding(
            namespace=namespace_name,
            text=EMBEDDING_TEXT,
            persist_source=True,
            create_namespace=True,
            embedding_model=embedding_model
        )
        assert response.status_code == status.HTTP_200_OK
        self.namespaces.append(namespace_name)

    def test_create_non_existing_namespace_no_create(self):
        namespace_name = generate_namespace_name()
        embedding_model = str(CloudflareEmbeddingModels.BAAISmall)

        response = create_cloudflare_embedding(
            namespace=namespace_name,
            text=EMBEDDING_TEXT,
            persist_source=True,
            create_namespace=False,
            embedding_model=embedding_model
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

        response_error_message = response.json().get("detail")[0].get("msg")
        assert bool(re.match(r"vector index with name", response_error_message.lower()))

    def test_create_cloudflare_embedding_payload(self):
        namespace_name = generate_namespace_name()

        embedding_payload = {
            "a": 1,
            "b": 2
        }
        response = create_cloudflare_embedding(
            namespace=namespace_name,
            text=EMBEDDING_TEXT,
            persist_source=True,
            create_namespace=True,
            payload=embedding_payload
        )
        assert response.status_code == status.HTTP_200_OK

        create_response_json = response.json()
        created_embedding_item = create_response_json.get("items", [])[0]

        response = client.get(
            url=CLOUDFLARE_EMBEDDING_PATH.format(
                namespace=namespace_name,
                embedding_id=created_embedding_item.get("id")
            ),
        )
        assert response.json().get("payload") == embedding_payload
        self.namespaces.append(namespace_name)

    def test_create_source(self):
        namespace_name = generate_namespace_name()
        response = create_cloudflare_embedding(
            namespace=namespace_name,
            text=EMBEDDING_TEXT,
            persist_source=True,
            create_namespace=True,
        )
        assert response.status_code == status.HTTP_200_OK

        create_response_json = response.json()
        created_embedding_json = create_response_json.get("items", [])[0]

        response = client.get(
            url=CLOUDFLARE_EMBEDDING_PATH.format(
                namespace=namespace_name,
                embedding_id=created_embedding_json.get("id")
            ),
        )
        assert response.json().get("source") == EMBEDDING_TEXT
        self.namespaces.append(namespace_name)

    def test_delete(self):
        namespace_name = generate_namespace_name()
        response = create_cloudflare_embedding(
            namespace=namespace_name,
            text=EMBEDDING_TEXT,
            persist_source=True,
            create_namespace=True,
        )
        assert response.status_code == status.HTTP_200_OK

        create_response_json = response.json()
        created_embedding_item = create_response_json.get("items", [])[0]

        response = client.delete(
            url=CLOUDFLARE_EMBEDDING_PATH.format(
                namespace=namespace_name,
                embedding_id=created_embedding_item.get("id")
            )
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json().get("success")
        self.namespaces.append(namespace_name)


class TestCloudflareNamespace(TestBase):
    def test_create(self):
        namespace_name = generate_namespace_name()
        response = create_cloudflare_namespace(
            name=namespace_name,
            preset=str(CloudflareEmbeddingModels.BAAISmall)
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json().get("name") == namespace_name
        assert response.json().get("dimensionality") == CloudflareEmbeddingModels.BAAISmall.dimensionality

        self.namespaces.append(namespace_name)

    def test_create_invalid_preset(self):
        namespace_name = generate_namespace_name()
        response = create_cloudflare_namespace(
            name=namespace_name,
            preset="invalid_namespace_name"
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        response_type = response.json().get('detail')[-1].get('type')
        assert response_type == "literal_error"
        self.namespaces.append(namespace_name)

    def test_create_default_preset(self):
        namespace_name = generate_namespace_name()
        response = create_cloudflare_namespace(
            name=namespace_name,
        )
        # check dimensionality
        dimensionality = CloudflareEmbeddingModels.BAAIBase.dimensionality
        assert response.json().get("dimensionality") == dimensionality
        self.namespaces.append(namespace_name)

    def test_delete_cloudflare_namespace(self):
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

        response = client.delete(
            url=DELETE_CLOUDFLARE_NAMESPACE_PATH.format(
                namespace=namespace_name
            )
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json().get("success")
        self.namespaces.append(namespace_name)

