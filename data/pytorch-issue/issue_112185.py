import torch

a = torch.randn(5, 3)
b = torch.randn(4, 3)

# eager mode shapes:
# x.shape: [B, * , D] -> [2, *, 3]
# x._values.shape: [sum(*), D] -> [9, 3]
# x._offsets.shape: [B]
x = jagged_from_list([a, b], None)

# Here are the expected faketensor shapes on the first run:
# first run: automatic dynamic shapes assumes everything is static by default
# (except that JaggedTensor is special, and the * dim is always treated as dynamic)
# x.shape: [2, s0, 3]
# x._values.shape: [9, 3]
# x._offsets.shape: [2]
out = compiled_fn(x)

x2 = x = jagged_from_list([b, a, b], None)

# Second run: we see that the batch size varied, so we mark it dynamic.
# We also see that the sum(*) dim on _values varied since it depends on the batch dim, so we mark it dynamic (I'm not actually sure if this is backed or unbacked)
# x2.shape: [s1, s0, 3]
# x2._values.shape: [s2, 3]
# x2._offsets.shape: [s1]
out = compiled_fn(2)

a = torch.randn(5, 3)
b = torch.randn(4, 3)
x = jagged_from_list([a, b], None)

# This is the batch dim on the JaggedTensor. We this should cause us to realize that:
# (1) x._offsets.shape[0] is x.shape[0] (they both use the same symbol), so x._offsets.shape[0] is dynamic 
# (2) x._values.shape[0] depends directly on the batch dim (even though it gets its on symbol), so it should also be dynamic
torch._dynamo.mark_dynamic(x, 0)

# Here are the expected faketensor shapes on the first run:
# 3 unique symbols, corresponding to B, *, and sum(*). `mark_dynamic()` usage above tells us that **both** B and sum(*) are dynamic.
# x.shape: [s0, s1, 3]
# x._values.shape: [s2, 3]
# x._offsets.shape: [s0]
out = compiled_fn(x)