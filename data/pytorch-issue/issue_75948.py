import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.backends.cudnn.deterministic = True

    # setup
    x = torch.rand(2, 4)
    layer = nn.Linear(4, 4)
    actions = torch.tensor([1, 2])

    # whole batch
    layer.weight.grad = None
    logprobs = Categorical(logits=layer(x)).log_prob(actions)
    logprobs.mean().backward()
    print(layer.weight.grad.sum())

    # gradient accumulation
    layer.weight.grad = None
    Categorical(logits=layer(x[0])).log_prob(actions[0]).mean().backward()
    Categorical(logits=layer(x[1])).log_prob(actions[1]).mean().backward()
    print(layer.weight.grad.sum() / 2)

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.backends.cudnn.deterministic = True

    # setup
    x = torch.rand(4, 4)
    layer = nn.Linear(4, 4)
    actions = torch.tensor([1, 2, 3, 0])

    # whole batch
    layer.weight.grad = None
    logprobs = Categorical(logits=layer(x)).log_prob(actions)
    logprobs.mean().backward()
    print(layer.weight.grad.sum())
    print(layer.weight.grad)

    # gradient accumulation
    layer.weight.grad = None
    probs0 = Categorical(logits=layer(x[0:2])).log_prob(actions[0:2])
    probs0.mean().backward()
    probs1 = Categorical(logits=layer(x[2:4])).log_prob(actions[2:4])
    probs1.mean().backward()
    print(layer.weight.grad.sum() / 2)
    print(layer.weight.grad / 2)