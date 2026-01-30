import torch
import torch.nn as nn

torch.ops.load_library(torch_wrapper.__file__)

@torch._library.register_fake_class("torch_wrapper::Test")
class FakeTest:
    def __init__(
        self, 
        x: int) -> None:
        self.x = x

    @classmethod
    def __obj_unflatten__(cls, flattened_test):  
        return cls(**dict(flattened_test))
    
    def __len__(self):
        return 0
    
    def __setstate__(self, state_dict):
        self.x = state_dict["x"]

@torch.library.register_fake("torch_wrapper::add_constant")
def fake_add_constant(test, x):
    return x.new_empty(*x.shape)

test = torch.classes.torch_wrapper.Test(5)
x = torch.ones(2, 2)

class Mod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.ops.torch_wrapper.add_constant(test, x)

exported_program = torch.export.export(Mod(), args=(x,), strict=False)
print(exported_program)

output_path = torch._inductor.aoti_compile_and_package( # Fails here
        exported_program,
        package_path=os.path.join(os.getcwd(), "model.pt2"),
    )