def test_pass(x):
    # type: (Tensor) -> Tensor
    c = Context(1)

    with c as mult:
        pass

    x *= c.count
    return x

def test_early_return(x, c):
    # type: (Tensor, PersistentContext) -> Tensor

    with c as mult:
        y = x ** mult
        return y

def test_serial(x):
    # type: (Tensor) -> Tensor

    c = Context(1)

    with c as mult:
        y = x ** mult

    with c as mult:
        y *= mult

    return y

def test_nested(x):
    # type: (Tensor) -> Tensor

    c = Context(1)

    with c as m:
        with c as n:
            y = x ** n

        y *= m

    return y

VAR = EXPR
VAR.__enter__()
try:
    BLOCK
finally:
    VAR.__exit__()

def test_serial(x):
    # type: (Tensor) -> Tensor

    c = Context(1)

    with c as mult:
        y = x ** mult

    return y

with A() as a, A() as b:
            pass

VAR = EXPR
VAR.__enter__()
try:
    BLOCK
finally:
    VAR.__exit__()