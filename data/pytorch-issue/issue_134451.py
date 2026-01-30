import torch
import torch.nn as nn
import torch_tensorrt
from enum import IntEnum

class BoundingBox2DIndex(IntEnum):
    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        return 5

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
    

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self._mlp_states = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, BoundingBox2DIndex.size()),
        )

    def forward(self, x):
        agent_states = self._mlp_states(x)
        agent_states[..., BoundingBox2DIndex.POINT] = (
            agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        )
        agent_states[..., BoundingBox2DIndex.HEADING] = (
            agent_states[..., BoundingBox2DIndex.HEADING].tanh() * torch.pi
        )
        return agent_states

# Instantiate the model and prepare input
model = SimpleModel().cuda()
input_tensor = torch.randn(1, 10, dtype=torch.float32).cuda()

# Configure TensorRT options
enabled_precisions = {torch.float16, torch.float32}

compilation_kwargs = {
    "enabled_precisions": enabled_precisions,
    "debug": True,
    "dryrun": True,
    "workspace_size": 20 << 30,
    "min_block_size": 5,
    "torch_executed_ops": {},
}

# Compile the model with TensorRT
trt_mod = torch.compile(model.eval(), backend="tensorrt", options=compilation_kwargs)
out_trt = trt_mod(input_tensor)
print(out_trt)