# mypy: enable-error-code="unused-ignore"
import operator

import torch
from torch import Tensor
from typing_extensions import assert_type, reveal_type

x = torch.randn(3)
assert_type(x, Tensor)

i64: int = 2
f64: float = 3.14

# op(Tensor, Tensor)
assert_type(x + x, Tensor)
assert_type(x - x, Tensor)
assert_type(x * x, Tensor)
assert_type(x / x, Tensor)
assert_type(x % x, Tensor)
assert_type(x // x, Tensor)  # type: ignore[assert-type]
assert_type(x**x, Tensor)  # type: ignore[assert-type]
# comparisons
assert_type(x < x, Tensor)
assert_type(x > x, Tensor)
assert_type(x <= x, Tensor)
assert_type(x >= x, Tensor)
assert_type(x == x, Tensor)
assert_type(x != x, Tensor)

# op(Tensor, int)
assert_type(x + i64, Tensor)
assert_type(x - i64, Tensor)
assert_type(x * i64, Tensor)
assert_type(x / i64, Tensor)
assert_type(x % i64, Tensor)
assert_type(x // i64, Tensor)  # type: ignore[assert-type]
assert_type(x**i64, Tensor)  # type: ignore[assert-type]
assert_type(x < i64, Tensor)
assert_type(x > i64, Tensor)
assert_type(x <= i64, Tensor)
assert_type(x >= i64, Tensor)
assert_type(x == i64, Tensor)
assert_type(x != i64, Tensor)

# op(Tensor, float)
assert_type(x + f64, Tensor)
assert_type(x - f64, Tensor)
assert_type(x * f64, Tensor)
assert_type(x / f64, Tensor)
assert_type(x % f64, Tensor)
assert_type(x // f64, Tensor)  # type: ignore[assert-type]
assert_type(x**f64, Tensor)  # type: ignore[assert-type]
assert_type(x < f64, Tensor)
assert_type(x > f64, Tensor)
assert_type(x <= f64, Tensor)
assert_type(x >= f64, Tensor)
assert_type(x == f64, Tensor)
assert_type(x != f64, Tensor)

# op(int, Tensor)
assert_type(i64 + x, Tensor)
assert_type(i64 - x, Tensor)  # type: ignore[assert-type]
assert_type(i64 * x, Tensor)
assert_type(i64 / x, Tensor)  # type: ignore[assert-type]
assert_type(i64 % x, Tensor)  # type: ignore[assert-type]
assert_type(i64 // x, Tensor)  # type: ignore[assert-type]
assert_type(i64**x, Tensor)  # type: ignore[assert-type]
assert_type(i64 < x, Tensor)
assert_type(i64 > x, Tensor)
assert_type(i64 <= x, Tensor)
assert_type(i64 >= x, Tensor)
assert_type(i64 == x, Tensor)  # type: ignore[assert-type]
assert_type(i64 != x, Tensor)  # type: ignore[assert-type]

# op(float, Tensor)
assert_type(f64 + x, Tensor)
assert_type(f64 - x, Tensor)  # type: ignore[assert-type]
assert_type(f64 * x, Tensor)
assert_type(f64 / x, Tensor)  # type: ignore[assert-type]
assert_type(f64 % x, Tensor)  # type: ignore[assert-type]
assert_type(f64 // x, Tensor)  # type: ignore[assert-type]
assert_type(f64**x, Tensor)  # type: ignore[assert-type]
assert_type(f64 < x, Tensor)
assert_type(f64 > x, Tensor)
assert_type(f64 <= x, Tensor)
assert_type(f64 >= x, Tensor)
assert_type(f64 == x, Tensor)  # type: ignore[assert-type]
assert_type(f64 != x, Tensor)  # type: ignore[assert-type]


OPS = [
    operator.add,  # +
    operator.sub,  # -
    operator.mul,  # *
    operator.truediv,  # /
    operator.mod,  # %
    operator.floordiv,  # //
    operator.pow,  # **
    operator.le,  # <
    operator.gt,  # >
    operator.lt,  # <=
    operator.ge,  # >=
    operator.eq,  # ==
    operator.ne,  # !=
]

for rhs in [x, i64, f64]:
    for op in OPS:
        assert isinstance(op(x, rhs), Tensor)
for lhs in [x, i64, f64]:
    for op in OPS:
        assert isinstance(op(lhs, x), Tensor)