# src/retrieve/strategies.py

def select_k(query: str, max_tokens: int = 3000) -> int:
    """
    Heuristic:
    - Short queries (<=50 chars): k=3
    - Medium queries (<=150 chars): k=5
    - Long queries: reduce k so that k*avg_chunk_size <= token budget
    """
    length = len(query)
    if length <= 50:
        return 3
    if length <= 150:
        return 5
    # estimate chunk tokens ≈ 200 tokens each
    est_per = 200
    return max(1, min(10, max_tokens // est_per))
