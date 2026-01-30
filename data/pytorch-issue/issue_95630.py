import torch.nn as nn

import torch

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel

import torch._dynamo as dynamo
dynamo.config.verbose = True

model = BoringModel()
compiled_model = torch.compile(model)
trainer = Trainer(fast_dev_run=True)
trainer.fit(compiled_model)

import torch


class MyModel(torch.nn.Module):
    def forward(self):
        ...

    def training_step(self):
        self()


model = MyModel()
compiled_model = torch.compile(model)

model.forward = compiled_model.dynamo_ctx(model.forward)
model.training_step = compiled_model.dynamo_ctx(model.training_step)

model.training_step()