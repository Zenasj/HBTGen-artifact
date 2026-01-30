import torch

def nan_to_num(
    a: TensorLikeType,
    nan: Optional[NumberType] = 0.0,
    posinf: Optional[NumberType] = None,
    neginf: Optional[NumberType] = None,
) -> TensorLikeType:
    assert isinstance(a, TensorLike)

    if a.dtype == torch.bool:
        return clone(a)

    if posinf is None:
        posinf = prims.maximum_value(a.dtype)

    if neginf is None:
        neginf = prims.minimum_value(a.dtype)

    print("A:", isnan(a))
    print("nan:", nan)
    result = where(isnan(a), nan, a)

    is_neg = signbit(a)
    is_neginf = bitwise_and(isinf(a), is_neg)
    result = where(is_neginf, neginf, result)

    is_posinf = bitwise_and(isinf(a), bitwise_not(is_neg))
    result = where(is_posinf, posinf, result)
    return result