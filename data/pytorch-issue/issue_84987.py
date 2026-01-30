expected_backend_op_names: List[OperatorName] = (
        list(backend_indices[backend_key].index.keys()) + []
        if autograd_key is None
        else list(backend_indices[autograd_key].index.keys())
    )