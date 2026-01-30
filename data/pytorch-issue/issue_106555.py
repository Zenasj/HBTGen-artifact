import torch
import torch.nn as nn

DEVICE = torch.device("cuda:0")
NUM_FEATURES = 256

def bias_sigmoid_mul(x1, x2, bias):
    x2 = torch.sigmoid(x2 + bias)
    y = x1 * x2
    return y


bias_sigmoid_mul_jit = torch.compile(bias_sigmoid_mul)


class ModuleWithJit(nn.Module):
    def __init__(self):
        super(ModuleWithJit, self).__init__()
        self.linear_1 = nn.Linear(NUM_FEATURES, NUM_FEATURES, bias=True)
        self.linear_2 = nn.Linear(NUM_FEATURES, NUM_FEATURES, bias=False)
        self.linear_2_bias = nn.Parameter(torch.zeros(NUM_FEATURES))

    def forward(self, input_tensor):
        x1 = self.linear_1(input_tensor)
        x2 = self.linear_2(input_tensor)
        output = bias_sigmoid_mul_jit(x1, x2, self.linear_2_bias)
        return output


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.module_with_jit_1 = ModuleWithJit()
        self.module_with_jit_2 = ModuleWithJit()

    def forward(self, x, gradient_checkpointing: bool):
        if gradient_checkpointing:
            y = torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=True)
        else:
            y = self._forward(x)
        return y

    def _forward(self, x):
        x = x + self.module_with_jit_1(x)
        x = x + self.module_with_jit_2(x.transpose(-2, -3)).transpose(-2, -3)
        return x


def main():
    torch.cuda.set_device(device=DEVICE)

    torch.manual_seed(1234567890)
    model = Model()
    model.train()
    model.to(device=DEVICE)
    model_parameters = list(model.parameters())

    torch.manual_seed(1234567890)
    input_tensor = torch.randn(1, 128, 256, NUM_FEATURES).to(device=DEVICE)
    input_tensor.requires_grad = True
    target_tensor = torch.randn(1, 128, 256, NUM_FEATURES).to(dtype=input_tensor.dtype, device=DEVICE)

    for iteration in range(10):
        print(f"iteration={iteration}")
        for param in model_parameters:
            param.grad = None
        output_tensor = model(
            x=input_tensor.clone(),
            gradient_checkpointing=True,
        )
        loss = torch.mean(torch.abs(target_tensor - output_tensor))
        loss.backward()


if __name__ == "__main__":
    main()