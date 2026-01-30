import torch
import logging
import sys
import torch._dynamo

# torch._logging.set_logs(dynamo=logging.DEBUG, bytecode=True)
torch._dynamo.config.print_graph_breaks = True

import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(5, 5)
        self.dropout = torch.nn.Dropout()

    def _init_attention_seed(self):
        """
        This function sets a new seed for the attention layer to make dropout deterministic for both forward calls: 1
        normal forward call and 1 forward call in backward to recalculate activations.
        """

        # randomize seeds
        # use cuda generator if available
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            # GPU
            device_idx = torch.cuda.current_device()
            self.attention_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            self.attention_seed = int(torch.seed() % sys.maxsize)

        torch.manual_seed(self.attention_seed)

    def forward(self, x):
        self._init_attention_seed()
        return self.dropout(self.linear(x))

x = torch.randn(5, 5)

m = MyModel()
print(m(x))

opt_m = torch.compile(backend="eager")(m)
print(opt_m(x))

call_hasattr

TorchVariable

Generator.seed