py
def foo(bar, baz):
    print(f"bar={bar}, baz={baz}")

foo(baz="baz", *["bar"])

bar=bar, baz=baz