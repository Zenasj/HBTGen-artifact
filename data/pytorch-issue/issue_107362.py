import os
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

        channels = (3, 32, 64, 128)
        self.num_levels = len(channels)
        for i in range(1, self.num_levels):
            dla_layer = Tree(
                channels[i - 1],
                channels[i],
            )
            layer_name = f"level{i}"
            self.add_module(layer_name, dla_layer)

    def forward(self, x):
        outs = []
        for i in range(1, self.num_levels):
            x = getattr(self, "level{}".format(i))(x)
            outs.append(x)
                    
        return tuple(outs)

class Tree(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super(Tree, self).__init__()
        self.conv = nn.Conv2d(2 * out_channels, out_channels, 1, 1, bias=False)
        self.tree1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.tree2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x1 = self.tree1(x)
        identity = self.project(x) 
        x2 = self.tree2(identity)
        x = self.conv(torch.concat([x2, x1], 1))
        return x


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(rank=rank, world_size=world_size)


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    model = TestNet().cuda()
    ddp_model = DDP(model, device_ids=[rank]) # important
    model = torch.compile(ddp_model.module)
    setattr(ddp_model, 'module', model )

    with torch.no_grad():
        outputs = ddp_model(torch.randn(2, 3, 128, 128).cuda())

if __name__ == '__main__':
    demo_basic(0, 1)

model = ...
ddp_model = DDP(model, ...)
compiled_model = torch.compile(ddp_model, ...)
output = compiled_model(input, ...)

if context is not None and context.fw_metadata:
                original_output_start_index = context.fw_metadata.num_mutated_inputs

user_visible_outputs = {
                n.name
                for n in model_outputs[
                    original_output_start_index : original_output_start_index
                    + num_orig_model_outputs
                ]
                if isinstance(n, torch.fx.Node)
            }