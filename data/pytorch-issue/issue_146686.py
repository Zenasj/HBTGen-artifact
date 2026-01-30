import torch

# This function is equivalent to compute_contiguous() from TensorImpl.cpp
def is_contiguous(a: TensorLikeType) -> bool:
    """
    Tests whether a tensor is contiguous or not.

    Tensors are contiguous when they have no elements,
    one element, or when they have "nested" strides.
    """
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious
    logger.info(f"inside is_contiguous with size: {a.size()} stride: {a.stride()}")
    if guard_size_oblivious(a.numel() < 2):
        return True

    expected_stride = 1
    for x, y in reversed(tuple(zip(a.shape, a.stride()))):
        # Skips checking strides when a dimension has length 1
        if guard_size_oblivious(x == 1):
            continue
        
        logger.info(f"checking the following {y} != {expected_stride}")

        if guard_size_oblivious(y != expected_stride):
            return False
        expected_stride = expected_stride * x

    return True