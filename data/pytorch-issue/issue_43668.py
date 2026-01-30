sparse_params = []
for index, param in enumerate(params):
    if isinstance(param, dict):
        for d_index, d_param in enumerate(param.get("params", [])):
            if d_param.is_sparse:
                sparse_params.append([index, d_index])
    elif param.is_sparse:
        sparse_params.append(index)
if sparse_params:
    raise ValueError(
        f"Sparse params at indices {sparse_params}: SparseAdam requires dense parameter tensors"
    )