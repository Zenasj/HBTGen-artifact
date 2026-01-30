import torch
import torch.nn as nn

model = resnet18(num_classes=num_classes, pretrained=False)
model.to(cpu_device)
print(model)
# Make a copy of the model for layer fusion
fused_model = copy.deepcopy(model)

model.train()
# The model has to be switched to training mode before any layer fusion.
# Otherwise the quantization aware training will not work correctly.
fused_model.train()

# Fuse the model in place rather manually.
fused_model = torch.quantization.fuse_modules(
    fused_model, [["conv1", "bn1", "relu"]], inplace=True)
for module_name, module in fused_model.named_children():
    if "layer" in module_name:
        for basic_block_name, basic_block in module.named_children():
            torch.quantization.fuse_modules(
                basic_block, [["conv1", "bn1", "relu"], ["conv2", "bn2"]], inplace=True)
            for sub_block_name, sub_block in basic_block.named_children():
                if sub_block_name == "downsample":
                    torch.quantization.fuse_modules(
                        sub_block, [["0", "1"]], inplace=True)

def forward(self, x: Tensor) -> Tensor:
        ...
        out = self.relu(out)  # here
        ...
        out += identity
        out = self.relu(out)  # and here
        return out

class QuantizableBasicBlock(BasicBlock):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_relu = torch.nn.quantized.FloatFunctional()

def forward(self, x: Tensor) -> Tensor:
        ...
        out = self.relu(out)  # first one
        ...
        out = self.add_relu.add_relu(out, identity)  # here the "new" relu is called, not the original self.relu
        return out