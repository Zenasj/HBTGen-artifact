assert query.shape[-1] == key.shape[-1], f"query has embedding dimension {query.shape[-1]} which does n't match key embedding dimension: {key.shape[-1]}"
assert key.shape == value.shape