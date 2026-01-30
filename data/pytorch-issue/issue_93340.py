import torch

def fn(x):
    return torch.distributions.Categorical(probs=x).entropy()

opt_fn = torch._dynamo.optimize("eager")(fn)
x = torch.rand([4, 4])
print(opt_fn(x))

self.logits

lazy_property

Categorical.logits