from typing import Optional, Dict, Any

from app.config import settings


def source_key() -> str:
    return f"{settings.NAMESPACE.lower()}_original"


def merge_metadata(metadata: Optional[Dict[str, Any]], text: str):
    source = {
        source_key(): text
    }
    if metadata:
        return {**metadata, **source}
    else:
        return source
