import torch

# post-grad graph
where_2: "i64[512*s0, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_257, unsqueeze_3, full_default_1);  unsqueeze_3 = full_default_1 = None
scatter_upon_const_tensor: "f32[512*s0, 30000][30000, 1]cuda:0" = torch__inductor_fx_passes_post_grad_scatter_upon_const_tensor(shape = [mul_37, 30000], background_val = 0, dtype = torch.float32, dim = 1, selector = where_2, val = -1.0);  where_2 = None

# post_grad.py
def scatter_upon_const_tensor(
    match: Match, shape, background_val, dtype, dim, selector, val
):
    """
    Match the pattern of full+scatter into a pointwise.

    TODO: Right now the scatter value must be a scalar. But we could support it
    when it is a tensor as well.
    """
    from torch._inductor import metrics

    metrics.num_matches_for_scatter_upon_const_tensor += 1

    selector_loader = selector.make_loader()  # <-- errors because selector is a Tensor in this case