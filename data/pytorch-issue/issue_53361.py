def sample_inputs_linalg_det(op_info, device, dtype, requires_grad):
    kw = dict(device=device, dtype=dtype)
    inputs = [
        random_square_matrix_of_rank(S, S - 2, **kw),  # dim2_null
    ]
    # if you uncomment the following line, test_fn_gradgrad_linalg_det_cuda_float64 passes!
    #inputs[0] = inputs[0].cpu()
    for t in inputs:
        t.requires_grad = requires_grad
    return [SampleInput(t) for t in inputs]