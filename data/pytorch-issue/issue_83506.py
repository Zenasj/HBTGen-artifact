py
import torch

test = torch.library.Library("test", "DEF")
test_impl = torch.library.Library("test", "IMPL", "CompositeExplicitAutograd")

schema = "test(int[1]? dim) -> Tensor"

test.define(schema)
test_impl.impl("test", lambda dim=None: torch.empty(dim))

try:
    torch.ops.test.test(dim=2)
except RuntimeError as e:
    print(e)

# But the following works:
import torch

schema = "test_non_optional(int[1] dim) -> Tensor"

test.define(schema)
test_impl.impl("test_non_optional", lambda dim=None: torch.empty(dim))

try:
    torch.ops.test.test_non_optional(dim=2)
except RuntimeError as e:
    print(e)