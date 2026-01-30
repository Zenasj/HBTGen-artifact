import torch.nn as nn

import torch

def custom_add_direct(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

@torch.library.custom_op("mylib::custom_add", mutates_args=(),
                         device_types="cuda",
                         )
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return custom_add_direct(a,b)

@torch.library.register_fake("mylib::custom_add")
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(a)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        y = self.relu(x)
        x = self.fc2(torch.ops.mylib.custom_add(x, y))
        x = self.sigmoid(x)
        return x

with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model().to(device=device)
    example_inputs = (torch.randn(8, 10, device=device),)

    # Export the model                                                                               
    exported = torch.export.export(model, example_inputs)

    # Compile the model                                                                              
    output_path = torch._inductor.aoti_compile_and_package(
        exported,
        example_inputs,
        package_path="model.pt2",
    )