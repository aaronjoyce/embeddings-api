from enum import Enum


class CloudflareEmbeddingModels(Enum):
    BAAISmall = "@cf/baai/bge-small-en-v1.5"
    BAAIBase = "@cf/baai/bge-base-en-v1.5"
    BAAILarge = "@cf/baai/bge-large-en-v1.5"
