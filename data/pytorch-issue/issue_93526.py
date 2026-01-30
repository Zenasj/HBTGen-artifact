import torchvision
import torch
import torchdynamo

dtype = torch.float16
model = torchvision.models.resnet50()
model.to(dtype)
model.cuda()
model.eval()

example_inputs = torch.randn((32,3,224,224), device="cuda", dtype=dtype)

def simple_op():
    with torch.cuda.nvtx.range("model"):  # <-- This causes the BUG
        return model(example_inputs)

@torchdynamo.optimize("inductor")
def run_once_dynamo_ind():
    return simple_op()

run_once_dynamo_ind()