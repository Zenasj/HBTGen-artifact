def fn_simple(x):
    if x.is_inference():
        return x.sum()
    else:
        return x.min()