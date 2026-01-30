import torch
import torch.nn as nn

mp.spawn(
        self._train,
        nprocs=len(gpu_ids))

optimizer, remain_optimizer = get_optimizer(model, optimizer)
if isinstance(dist_config, DistConfig) \
            and dist_config.multiprocessing_distributed and dist_config.sync_bn:
      model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

def forward(self, *inputs):
        x, *other = inputs
        b1, b2, b3 = self.bone(x)
        out1, out2, out3 = self.head(b1, b2, b3)
        return [out1, out2, out3]

if isinstance(dist_config, DistConfig) \
            and dist_config.multiprocessing_distributed and dist_config.sync_bn:
      model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            print(id(group['params'][1])) # get the id of parameter in optimizer
            for p in group['params']:
                if p.grad is None:
                    continue

def forward(self, *inputs):
        x, *other = inputs
        b1, b2, b3 = self.bone(x)
        out1, out2, out3 = self.head(b1, b2, b3)
        id(list(self.parameters())[1])  # get the id of parameter in model
        return [out1, out2, out3]