def foo(x):
    # type: (int) -> int
    if isinstance(x, str):
        return x["1"]
    return x + 1

reveal_type(foo) # no error, shows int -> int