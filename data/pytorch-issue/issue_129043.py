import torch

def inner_fn(idx):
        selector_idx = list(idx)
        selector_idx[dim] = 0  # can do this since the index tensor has a single element on the scatter dimension

        selector = selector_loader(selector_idx)
        return ops.where(
            selector == ops.index_expr(idx[dim], torch.int64),
            ops.constant(val, dtype),
            ops.constant(background_val, dtype),
        )