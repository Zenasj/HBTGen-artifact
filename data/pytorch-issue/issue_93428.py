import torch

aten.copy_.default
(525312, 513, 1)
(787968, 513, 1)

opt_mod = torch._dynamo.optimize("aot_eager_decomp_partition")(mod)