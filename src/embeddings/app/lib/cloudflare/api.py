import json
import enum
import re
import traceback

import CloudFlare

from fastapi import status
from typing import Optional, List, Dict, Any

from retry import retry

from embeddings.app.lib.cloudflare.models import CreateDatabaseRecord

from embeddings.exceptions import EmbeddingDimensionalityException, NotFoundException

from .models import VectorPayloadItem


# Error codes
ERROR_CODE_VECTOR_INDEX_NOT_FOUND = 3000
ERROR_CODE_INSERT_VECTOR_INDEX_SIZE_MISMATCH = 4003


# input dimensions
MAX_EMBEDDING_INPUT_TOKENS = 512

# output dimensions
OUTPUT_SMALL_DIMENSION = 384
OUTPUT_BASE_DIMENSION = 768
OUTPUT_LARGE_DIMENSION = 1024


MODEL_NAME_BGE_SMALL = "@cf/baai/bge-small-en-v1.5"
MODEL_NAME_BGE_BASE = "@cf/baai/bge-base-en-v1.5"
MODEL_NAME_BGE_LARGE = "@cf/baai/bge-large-en-v1.5"


MODEL_OUTPUT_DIMENSIONS = {
    MODEL_NAME_BGE_SMALL: OUTPUT_SMALL_DIMENSION,
    MODEL_NAME_BGE_BASE: OUTPUT_BASE_DIMENSION,
    MODEL_NAME_BGE_LARGE: OUTPUT_LARGE_DIMENSION
}


class CloudflareEmbeddingModels(enum.Enum):
    BAAISmall = MODEL_NAME_BGE_SMALL
    BAAIBase = MODEL_NAME_BGE_BASE
    BAAILarge = MODEL_NAME_BGE_LARGE

    def __str__(self) -> str:
        return str(self.value)

    @property
    def dimensionality(self) -> int:
        return MODEL_OUTPUT_DIMENSIONS.get(self.value)


DIMENSIONALITY_PRESETS = {
    OUTPUT_SMALL_DIMENSION: [CloudflareEmbeddingModels.BAAISmall],
    OUTPUT_BASE_DIMENSION: [CloudflareEmbeddingModels.BAAIBase],
    OUTPUT_LARGE_DIMENSION: [CloudflareEmbeddingModels.BAAILarge]
}


class API:

    def __init__(self, api_token: str, account_id: str):
        self.api_token = api_token
        self.account_id = account_id
        self.client = CloudFlare.CloudFlare(token=self.api_token)

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def create_vector_index(self, name: str, preset: str, description: Optional[str] = None):
        data = {
            "name": name,
            "config": {
                "preset": preset
            }
        }
        if description is not None:
            data["description"] = description

        res = self.client.accounts.vectorize.indexes.post(
            self.account_id,
            data=data
        )
        return res

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def query_vector_index(
        self,
        vector_index_name: str,
        vector: List[float],
        top_k: Optional[int] = 5,
        return_vectors: Optional[bool] = False,
        return_metadata: Optional[bool] = False,
        metadata_filter: Optional[Dict[str, Any]] = None
    ):
        data = {
            "vector": vector,
            "topK": top_k,
            "returnMetadata": return_metadata,
        }
        if return_vectors:
            data["returnValues"] = True

        if metadata_filter is not None:
            data["filter"] = metadata_filter

        res = self.client.accounts.vectorize.indexes.query.post(
            self.account_id,
            vector_index_name,
            data=data
        )
        return res

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def list_vector_indexes(self):
        res = self.client.accounts.vectorize.indexes(
            self.account_id
        )
        return res

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def vectors_by_ids(self, vector_index_name: str, ids: List[str]):
        res = self.client.accounts.vectorize.indexes.get_by_ids.post(
            self.account_id,
            vector_index_name,
            data={
                "ids": ids
            }
        )
        return res

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def vector_index_by_name(self, name: str):
        try:
            res = self.client.accounts.vectorize.indexes(
                self.account_id,
                name
            )
        except CloudFlare.exceptions.CloudFlareAPIError as ex:
            raise ex

        return res

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def delete_vector_index_by_name(self, name: str):
        try:
            res = self.client.accounts.vectorize.indexes.delete(
                self.account_id,
                name
            )
        except CloudFlare.exceptions.CloudFlareAPIError as ex:
            raise ex

        return res

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def delete_vectors_by_ids(self, vector_index_name: str, ids: List[str]):
        try:
            res = self.client.accounts.vectorize.indexes.delete_by_ids.post(
                self.account_id,
                vector_index_name,
                data={
                    "ids": ids
                }
            )
        except CloudFlare.exceptions.CloudFlareAPIError as ex:
            raise ex

        return res

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def insert_vectors(
            self,
            vector_index_name: str,
            vectors: List[VectorPayloadItem],
            create_on_not_found: bool = False,
            model_name: CloudflareEmbeddingModels = None
    ):
        data = "\n".join([json.dumps({"id": o.id, "values": o.values, "metadata": o.metadata}) for o in vectors])
        try:
            res = self.client.accounts.vectorize.indexes.insert.post(
                self.account_id,
                vector_index_name,
                data=data
            )
        except CloudFlare.exceptions.CloudFlareAPIError as ex:
            exception_status_code = int(ex)
            if exception_status_code == ERROR_CODE_INSERT_VECTOR_INDEX_SIZE_MISMATCH:
                matches = re.search(
                    r"the vector length is incorrect for this index; must be (\d+), got (\d+)",
                    str(ex)
                )
                expected_dimension = int(matches.group(1))
                received_dimension = int(matches.group(2))
                compatible_model_names = ','.join([str(o) for o in DIMENSIONALITY_PRESETS.get(expected_dimension, [])])
                # raise a pydantic validation error?
                raise EmbeddingDimensionalityException(
                    f"The embedding model's dimensionality: {received_dimension} is "
                    f"not compatible with the dimensionality of the namespace '{vector_index_name}', "
                    f"dimensionality: {expected_dimension}. "
                    f"Please provide one of the following compatible models: {compatible_model_names}",
                )

            elif exception_status_code == ERROR_CODE_VECTOR_INDEX_NOT_FOUND:
                if create_on_not_found:
                    # infer dimensionality from the vector at index 0
                    default_dimensionality_presets = DIMENSIONALITY_PRESETS.get(len(vectors[0].values), [])
                    if not default_dimensionality_presets:
                        allowed_dimensionality_values = ','.join([str(o) for o in DIMENSIONALITY_PRESETS.keys()])
                        raise Exception(
                            f"Unsupported vector preset dimensionality. "
                            f"Expected one of: {','.join(allowed_dimensionality_values)}, got: {len(vectors[0].values)}"
                        )

                    preset = str(model_name) if model_name is not None else default_dimensionality_presets[0].value
                    self.create_vector_index(
                        name=vector_index_name,
                        preset=preset
                    )
                    return self.insert_vectors(
                        vector_index_name=vector_index_name,
                        vectors=vectors
                    )
                else:
                    raise NotFoundException(
                        f"Vector index with name '{vector_index_name}' not found. "
                        f"Create the index via a separate call or include 'create_namespace' "
                        f"in your payload to automagically create and insert."
                    )

            raise ex

        return res

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def embed(self, model, texts: List[str]):
        res = self.client.accounts.ai.run.post(
            self.account_id,
            model,
            data={
                "text": texts
            }
        )
        return res

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def create_database_table(self, database_id: str, table_name: str):
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                vector_id TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """
        res = self.client.accounts.d1.database.query.post(
            self.account_id,
            database_id,
            data={
                "sql": create_table_sql
            }
        )
        return res

    def upsert_database_table_records(self, database_id: str, table_name: str, records: List[CreateDatabaseRecord]):
        formatted_records = [f"('{record.source}', '{record.vector_id}')" for record in records]
        formatted_insertion_query = ",".join(formatted_records)
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                vector_id TEXT NOT NULL UNIQUE,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """
        insertion_sql = f"""
            INSERT INTO {table_name} (source, vector_id) 
            VALUES {formatted_insertion_query} 
            ON CONFLICT (vector_id) DO UPDATE SET source = EXCLUDED.source;
        """
        res = self.client.accounts.d1.database.query.post(
            self.account_id,
            database_id,
            data={
                "sql": f"{create_table_sql} {insertion_sql}"
            }
        )
        return res[-1]

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def database_table_records_by_vector_ids(self, database_id: str, table_name: str, vector_ids: List[str]):
        formatted_ids = ",".join(["'{}'".format(o) for o in vector_ids])
        sql = f"""SELECT source, vector_id FROM {table_name} WHERE vector_id IN ({formatted_ids});"""
        res = self.client.accounts.d1.database.query.post(
            self.account_id,
            database_id,
            data={
                "sql": sql
            }
        )
        return res

    @retry(tries=5, delay=1, backoff=1, jitter=0.5)
    def list_database_table_records(
            self,
            database_id: str,
            table_name: str,
            limit: int = 20,
            offset: int = 0
    ):
        sql = f"SELECT source, vector_id FROM {table_name} LIMIT {limit} OFFSET {offset};"
        res = self.client.accounts.d1.database.query.post(
            self.account_id,
            database_id,
            data={
                "sql": sql
            }
        )
        return res
