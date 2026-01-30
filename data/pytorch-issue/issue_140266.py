from torch.distributions import Categorical
import torch

class SubCateg(Categorical):
    ...
    # probs = Categorical.probs  # uncomment to solve bug

@torch.compile()
def make_dist_and_execute(t, d):
    categ = d(logits=t)
    # just access stuff from the distribution to check everything runs fine
    a = categ.log_prob(categ.sample()) + categ.probs + categ.logits
    return a

for _ in range(2):
    make_dist_and_execute(torch.randn(10), SubCateg)