import tensorflow as tf

reveal_type(tf.Tensor)

import tensorflow as tf


class A(tf.Module):
    pass


class B(A):
    def foo(self) -> str:
        return "foo"


class C(B):
    def __init__(self, value: int) -> None:
        super().__init__()
        self.value = value


class D1(C):
    pass


class D2(C):
    def __init__(self, value: int) -> None:
        super().__init__(value)

    def do_something(self) -> int:
        return self.value


class D3(C):
    def do_something(self) -> int:
        return self.value


reveal_type(A())  # Reveals 'A' <- OK
reveal_type(B())  # Reveals 'B' <- OK
reveal_type(B().foo)  # Reveals 'str' <- OK
reveal_type(C(1))  # Reveals 'C' <- OK
reveal_type(D1(1))  # Reveals 'C' <- ??? 'D1' expected
reveal_type(D2(1))  # Reveals 'D2' <- OK but why?
reveal_type(D2(1).do_something)  # Reveals 'int` <- OK
reveal_type(D3(1))  # Reveals 'C' <- ??? 'D3' expected
reveal_type(D3(1).do_something)   # Reveals 'Any' <- ??? 'int' expected