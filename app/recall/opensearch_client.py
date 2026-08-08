from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

from app.utils.path_utils import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def get_opensearch_client() -> Any:
    from opensearchpy import OpenSearch

    host = os.getenv("OPENSEARCH_HOST", "localhost").strip()
    port = int(os.getenv("OPENSEARCH_PORT", "9200"))
    user = os.getenv("OPENSEARCH_USER", "admin").strip()
    password = os.getenv("OPENSEARCH_PASS", "admin")
    kwargs: dict[str, Any] = {
        "hosts": [{"host": host, "port": port}],
        "use_ssl": False,
        "timeout": 30,
    }
    if user:
        kwargs["http_auth"] = (user, password)
    return OpenSearch(**kwargs)


__all__ = ["get_opensearch_client"]
