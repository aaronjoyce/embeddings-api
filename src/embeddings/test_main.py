import re
import time

from typing import Dict, Any, List
from fastapi import status
from fastapi.testclient import TestClient

from embeddings.app.lib.cloudflare.api import CloudflareEmbeddingModels

from .main import app

client = TestClient(app)


CLOUDFLARE_NAMESPACE_PATH = "/api/v1/namespace/cloudflare"
CREATE_CLOUDFLARE_EMBEDDING_PATH = "/api/v1/embeddings/cloudflare"
RETRIEVE_DELETE_CLOUDFLARE_EMBEDDING_PATH = "/api/v1/embeddings/cloudflare/{namespace}/{embedding_id}"
DELETE_CLOUDFLARE_NAMESPACE_PATH = "/api/v1/namespace/cloudflare/{namespace}"
QUERY_CLOUDFLARE_NAMESPACE_PATH = "/api/v1/namespace/cloudflare/{namespace}/query"

QDRANT_NAMESPACE_PATH = "/api/v1/namespace/qdrant"
CREATE_QDRANT_EMBEDDING_PATH = "/api/v1/embeddings/qdrant"
RETRIEVE_DELETE_QDRANT_EMBEDDING_PATH = "/api/v1/embeddings/qdrant/{namespace}/{embedding_id}"
DELETE_QDRANT_NAMESPACE_PATH = "/api/v1/namespace/qdrant/{namespace}"
QUERY_QDRANT_NAMESPACE_PATH = "/api/v1/namespace/qdrant/{namespace}/query"


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
            url=CLOUDFLARE_NAMESPACE_PATH,
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


def create_qdrant_namespace(name: str, dimensionality: int = 768, distance: str = None):
    data = {
        "dimensionality": dimensionality,
        "name": name,
    }
    if distance is not None:
        data["distance"] = distance

    return client.post(
        url=QDRANT_NAMESPACE_PATH,
        json=data
    )


def create_cloudflare_namespace(name: str, preset: str = None):
    data = {
        "name": name
    }
    if preset is not None:
        data["preset"] = preset

    response = client.post(
        url=CLOUDFLARE_NAMESPACE_PATH,
        json=data
    )
    return response


def create_qdrant_embedding(
    namespace: str,
    text: str,
    persist_original: bool = True,
    create_namespace: bool = True,
    embedding_model: str = None,
    payload: Dict[str, Any] = None
):
    data = {
        "inputs": [{
            "text": text,
            "persist_original": persist_original,
            "payload": payload if payload else None,
        }],
        "create_namespace": create_namespace
    }
    if embedding_model is not None:
        data["embedding_model"] = embedding_model

    response = client.post(
        url=f"{CREATE_QDRANT_EMBEDDING_PATH}/{namespace}",
        json=data
    )
    return response


def create_cloudflare_embedding(
        namespace: str,
        text: str,
        persist_original: bool = True,
        create_namespace: bool = True,
        embedding_model: str = None,
        payload: Dict[str, Any] = None,
):
    data = {
        "inputs": [{
            "text": text,
            "persist_original": persist_original,
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


def query_cloudflare_namespace(
    text: str,
    namespace: str,
    metadata_filter: Dict[str, Any] = None
):
    data = {
        "inputs": text,
        "return_vectors": True,
        "return_metadata": True,
        "limit": 1,
    }
    if metadata_filter is not None:
        data["filter"] = metadata_filter

    response = client.post(
        url=QUERY_CLOUDFLARE_NAMESPACE_PATH.format(
            namespace=namespace
        ),
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
        assert response.status_code == status.HTTP_201_CREATED

        response = create_cloudflare_embedding(
            namespace=namespace_name,
            text=EMBEDDING_TEXT,
            persist_original=True,
            create_namespace=False,
            embedding_model=embedding_model
        )
        assert response.status_code == status.HTTP_201_CREATED
        self.namespaces.append(namespace_name)

    def test_create_non_existing_namespace_auto_create(self):
        namespace_name = generate_namespace_name()
        embedding_model = str(CloudflareEmbeddingModels.BAAISmall)
        response = create_cloudflare_embedding(
            namespace=namespace_name,
            text=EMBEDDING_TEXT,
            persist_original=True,
            create_namespace=True,
            embedding_model=embedding_model
        )
        assert response.status_code == status.HTTP_201_CREATED
        self.namespaces.append(namespace_name)

    def test_create_non_existing_namespace_no_create(self):
        namespace_name = generate_namespace_name()
        embedding_model = str(CloudflareEmbeddingModels.BAAISmall)

        response = create_cloudflare_embedding(
            namespace=namespace_name,
            text=EMBEDDING_TEXT,
            persist_original=True,
            create_namespace=False,
            embedding_model=embedding_model
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

        response_error_message = response.json().get("msg")
        assert bool(re.match(r"vector index with name", response_error_message.lower()))

    def test_create_payload(self):
        namespace_name = generate_namespace_name()

        embedding_payload = {
            "a": 1,
            "b": 2
        }
        response = create_cloudflare_embedding(
            namespace=namespace_name,
            text=EMBEDDING_TEXT,
            persist_original=True,
            create_namespace=True,
            payload=embedding_payload
        )
        assert response.status_code == status.HTTP_201_CREATED

        create_response_json = response.json()
        created_embedding_item = create_response_json.get("items", [])[0]

        response = client.get(
            url=RETRIEVE_DELETE_CLOUDFLARE_EMBEDDING_PATH.format(
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
            persist_original=True,
            create_namespace=True,
        )
        assert response.status_code == status.HTTP_201_CREATED

        create_response_json = response.json()
        created_embedding_json = create_response_json.get("items", [])[0]

        response = client.get(
            url=RETRIEVE_DELETE_CLOUDFLARE_EMBEDDING_PATH.format(
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
            persist_original=True,
            create_namespace=True,
        )
        assert response.status_code == status.HTTP_201_CREATED

        create_response_json = response.json()
        created_embedding_item = create_response_json.get("items", [])[0]

        response = client.delete(
            url=RETRIEVE_DELETE_CLOUDFLARE_EMBEDDING_PATH.format(
                namespace=namespace_name,
                embedding_id=created_embedding_item.get("id")
            )
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json().get("success")


class TestCloudflareNamespace(TestBase):

    def test_create(self):
        namespace_name = generate_namespace_name()
        response = create_cloudflare_namespace(
            name=namespace_name,
            preset=str(CloudflareEmbeddingModels.BAAISmall)
        )
        assert response.status_code == status.HTTP_201_CREATED
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

    def test_delete_namespace(self):
        namespace_name = generate_namespace_name()

        response = client.post(
            url=CLOUDFLARE_NAMESPACE_PATH,
            json={
                "preset": str(CloudflareEmbeddingModels.BAAISmall),
                "name": namespace_name
            }
        )
        assert response.status_code == status.HTTP_201_CREATED
        assert response.json().get("name") == namespace_name

        response = client.delete(
            url=DELETE_CLOUDFLARE_NAMESPACE_PATH.format(
                namespace=namespace_name
            )
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json().get("success")
        self.namespaces.append(namespace_name)

    def test_list(self):
        namespace_name = generate_namespace_name()

        response = client.post(
            url=CLOUDFLARE_NAMESPACE_PATH,
            json={
                "preset": str(CloudflareEmbeddingModels.BAAISmall),
                "name": namespace_name
            }
        )
        assert response.status_code == status.HTTP_201_CREATED
        assert response.json().get("name") == namespace_name

        response = client.get(
            url=CLOUDFLARE_NAMESPACE_PATH,
        )
        assert any([o.get('name') == namespace_name for o in response.json().get('items', [])])
        self.namespaces.append(namespace_name)


class TestCloudflareEmbeddingQuery(TestBase):

    def setup_method(self):
        super().setup_method()

        namespace = generate_namespace_name()
        # write data to the vector store
        response = create_cloudflare_embedding(
            namespace=namespace,
            text=EMBEDDING_TEXT,
            persist_original=True,
            create_namespace=True,
            payload={
                'a': 1,
                'b': 2
            }
        )
        self.embedding_id = response.json().get("items", [])[0].get("id")
        self.namespaces.append(namespace)

    def test_query_match(self):
        namespace = self.namespaces[0]

        response = query_cloudflare_namespace(
            namespace=namespace,
            text=EMBEDDING_TEXT,
        )
        try:
            assert response.status_code == status.HTTP_200_OK
        except Exception as ex:
            raise Exception(response.text)

        assert any([o.get("id") == self.embedding_id for o in response.json().get('items', [])])

    def test_query_metadata_filter_match(self):
        namespace = self.namespaces[0]
        response = query_cloudflare_namespace(
            namespace=namespace,
            text=EMBEDDING_TEXT,
            metadata_filter={
                "a": 1
            }
        )
        try:
            assert response.status_code == status.HTTP_200_OK
        except Exception as ex:
            raise Exception(response.text)

        assert any([o.get("id") == self.embedding_id for o in response.json().get('items', [])])

    def test_query_metadata_filter_not_match(self):
        namespace = self.namespaces[0]
        response = query_cloudflare_namespace(
            namespace=namespace,
            text=EMBEDDING_TEXT,
            metadata_filter={
                "a": 2
            }
        )
        try:
            assert response.status_code == status.HTTP_200_OK
        except Exception as ex:
            raise Exception(response.text)

        assert not any([o.get("id") == self.embedding_id for o in response.json().get('items', [])])


class TestQdrantBase:

    namespaces: List[str] = []

    def setup_method(self):
        self.namespaces = []

    def teardown_method(self):
        for namespace in self.namespaces:
            client.delete(
                url=DELETE_QDRANT_NAMESPACE_PATH.format(
                    namespace=namespace
                )
            )


class TestQdrantNamespace(TestQdrantBase):

    def test_create(self):
        namespace_name = generate_namespace_name()
        response = create_qdrant_namespace(
            name=namespace_name,
        )
        assert response.status_code == status.HTTP_201_CREATED
        assert response.json().get("name") == namespace_name
        self.namespaces.append(namespace_name)

    def test_create_dimensionality(self):
        namespace_name = generate_namespace_name()
        dimensionality = 512
        response = create_qdrant_namespace(
            name=namespace_name,
            dimensionality=dimensionality
        )
        assert response.status_code == status.HTTP_201_CREATED
        assert response.json().get("dimensionality") == dimensionality

        self.namespaces.append(namespace_name)

    def test_create_distance(self):
        namespace_name = generate_namespace_name()
        distance = "Euclid"
        response = create_qdrant_namespace(
            name=namespace_name,
            distance=distance
        )
        assert response.status_code == status.HTTP_201_CREATED
        assert response.json().get("distance") == distance

        self.namespaces.append(namespace_name)

    def test_delete(self):
        namespace_name = generate_namespace_name()
        create_qdrant_namespace(
            name=namespace_name,
        )
        delete_response = client.delete(
            url=DELETE_QDRANT_NAMESPACE_PATH.format(
                namespace=namespace_name
            ),
        )
        assert delete_response.status_code == status.HTTP_200_OK

    def test_list(self):
        namespace_name = generate_namespace_name()

        response = client.post(
            url=QDRANT_NAMESPACE_PATH,
            json={
                "name": namespace_name
            }
        )
        assert response.status_code == status.HTTP_201_CREATED

        response = client.get(
            url=QDRANT_NAMESPACE_PATH,
        )
        assert any([o.get('name') == namespace_name for o in response.json().get('items')])


class TestQdrantEmbedding(TestQdrantBase):

    def test_create_existing_namespace(self):
        # first, create a namespace
        namespace_name = generate_namespace_name()

        response = create_qdrant_namespace(
            name=namespace_name
        )
        assert response.status_code == status.HTTP_201_CREATED

        response = create_qdrant_embedding(
            namespace=namespace_name,
            text=EMBEDDING_TEXT,
            persist_original=True,
            create_namespace=False,
        )
        assert response.status_code == status.HTTP_201_CREATED
        self.namespaces.append(namespace_name)

    def test_create_non_existing_namespace_auto_create(self):
        namespace_name = generate_namespace_name()
        response = create_qdrant_embedding(
            namespace=namespace_name,
            text=EMBEDDING_TEXT,
            persist_original=True,
            create_namespace=True,
        )
        assert response.status_code == status.HTTP_201_CREATED
        self.namespaces.append(namespace_name)

    def test_create_non_existing_namespace_no_create(self):
        namespace_name = generate_namespace_name()
        response = create_qdrant_embedding(
            namespace=namespace_name,
            text=EMBEDDING_TEXT,
            persist_original=True,
            create_namespace=False,
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

        response_error_message = response.json().get("msg")
        assert bool(re.match(r"collection with name", response_error_message.lower()))

    def test_create_payload(self):
        namespace_name = generate_namespace_name()

        embedding_payload = {
            "a": 1,
            "b": 2
        }
        response = create_qdrant_embedding(
            namespace=namespace_name,
            text=EMBEDDING_TEXT,
            persist_original=True,
            create_namespace=True,
            payload=embedding_payload
        )
        assert response.status_code == status.HTTP_201_CREATED
        created_embedding_item = response.json().get("items", [])[0]

        response = client.get(
            url=RETRIEVE_DELETE_QDRANT_EMBEDDING_PATH.format(
                namespace=namespace_name,
                embedding_id=created_embedding_item.get("id")
            ),
        )
        assert response.json().get("payload") == embedding_payload
        self.namespaces.append(namespace_name)

    def test_create_source(self):
        namespace_name = generate_namespace_name()
        response = create_qdrant_embedding(
            namespace=namespace_name,
            text=EMBEDDING_TEXT,
            persist_original=True,
            create_namespace=True,
        )
        assert response.status_code == status.HTTP_201_CREATED
        created_embedding_json = response.json().get("items", [])[0]

        response = client.get(
            url=RETRIEVE_DELETE_QDRANT_EMBEDDING_PATH.format(
                namespace=namespace_name,
                embedding_id=created_embedding_json.get("id")
            ),
        )
        assert response.json().get("source") == EMBEDDING_TEXT
        self.namespaces.append(namespace_name)

    def test_delete(self):
        namespace_name = generate_namespace_name()
        response = create_qdrant_embedding(
            namespace=namespace_name,
            text=EMBEDDING_TEXT,
            persist_original=True,
            create_namespace=True,
        )
        assert response.status_code == status.HTTP_201_CREATED
        created_embedding_item = response.json().get("items", [])[0]

        response = client.delete(
            url=RETRIEVE_DELETE_QDRANT_EMBEDDING_PATH.format(
                namespace=namespace_name,
                embedding_id=created_embedding_item.get("id")
            )
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json().get("success")

