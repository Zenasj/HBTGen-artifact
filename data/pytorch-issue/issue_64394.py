import torch.nn as nn

class model:
    def _init(self):
        self.layer1 = FSDP(nn.linear())
        self.layer2 = FSDP(nn.linear())
        self.layer3 = nn.linear()
fsdp_model = FSDP(model())

class model:
    def _init(self):
        self.layer1 = nn.linear()
        self.layer2 = nn.linear()
        self.layer3 = nn.linear()
fsdp_model = FSDP(model(), 
                  auto_wrap_policy=functools.partial(default_auto_wrap_policy, min_num_params=1e7))