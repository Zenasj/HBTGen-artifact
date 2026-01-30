import torch
import torch._dynamo
torch._inductor.config.cpp.inject_relu_bug_TESTING_ONLY = 'accuracy'
torch._dynamo.config.repro_after = "aot"
torch._dynamo.config.repro_level = 4
torch._dynamo.config.debug_dir_root = "/tmp/tmp5qmh2x01"
@torch.compile()
def inner(x):
    for _ in range(3):
        x = torch.sin(x)
    x = torch.relu(x)
    for _ in range(3):
        x = torch.cos(x)
    return x

inner(torch.randn(20, 20).to("cpu"))