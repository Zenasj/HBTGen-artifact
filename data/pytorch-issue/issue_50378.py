kj
import torch as t
from torch.distributions import Categorical
probs = t.tensor([.3, .3, .399, .001])
y_test = t.tensor([0, 1, 2, 1, 2, 2, 3])
print(t.exp(Categorical(probs=probs).log_prob(y_test)))
print(t.exp(Categorical(logits=t.log(probs)).log_prob(y_test)))
print(t.exp(Categorical(logits=t.logit(probs)).log_prob(y_test)))

tensor([0.3000, 0.3000, 0.3990, 0.3000, 0.3990, 0.3990, 0.0010])
tensor([0.3000, 0.3000, 0.3990, 0.3000, 0.3990, 0.3990, 0.0010])
tensor([0.2816, 0.2816, 0.4362, 0.2816, 0.4362, 0.4362, 0.0007])