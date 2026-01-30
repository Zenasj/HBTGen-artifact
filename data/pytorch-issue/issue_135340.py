if fused:
    ...
    if foreach:
        raise RuntimeError("`fused` and `foreach` cannot be `True` together.")