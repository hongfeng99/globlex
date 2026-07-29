from app.resilience.circuit_breaker import (
    CircuitBreaker,
)


TOOL_BREAKERS = {
    name: CircuitBreaker(
        name,
        failure_threshold=threshold,
        recovery_timeout=timeout,
    )
    for name, threshold, timeout in [
        ("item_search_amazon", 0.30, 300),
        ("item_search_shopee", 0.30, 300),
        ("item_search_aliexpress", 0.30, 300),
        ("item_search_ebay", 0.30, 300),
        ("reranker", 0.20, 120),
        ("tower_encode", 0.20, 120),
    ]
}


def get_breaker(
    tool_name: str,
) -> CircuitBreaker | None:
    return TOOL_BREAKERS.get(tool_name)


__all__ = ["TOOL_BREAKERS", "get_breaker"]
