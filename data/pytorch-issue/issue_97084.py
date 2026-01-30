import torch, timm
import torch._dynamo

timm.models.layers.fast_norm.set_fast_norm(enable=False)
model = timm.create_model("convnext_tiny")
print(torch._dynamo.explain(model, torch.randn(1,3,224,224))[0])
# Dynamo produced 1 graphs with 0 graph break and 59 ops

timm.models.layers.fast_norm.set_fast_norm(enable=True)
model = timm.create_model("convnext_tiny")
print(torch._dynamo.explain(model, torch.randn(1,3,224,224))[0])
# Dynamo produced 49 graphs with 48 graph break and 36 ops

import torch, timm
import torch._dynamo

timm.models.layers.fast_norm.set_fast_norm(enable=True)
model = timm.create_model("convnext_tiny")
print(torch._dynamo.explain(model, torch.randn(1,3,224,224)).graph_break_count)