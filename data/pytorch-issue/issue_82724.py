from os import device_encoding
from turtle import forward
from unittest.mock import NonCallableMock
import torch
import torch.nn as nn

class Test(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor_span):
        # tensor_span shape: (batch, 1)
        batch_size = tensor_span.shape[0]
        test_mask = torch.zeros((batch_size, 100, 100), dtype=torch.float32, device=tensor_span.device)
        for i in range(batch_size):
            test_mask[i][:tensor_span[i, 0]][:tensor_span[i, 0]].fill_(1.0)
            # we can avoid the bug by indexing the tensor with [i, :tensor_span[i, 0], :tensor_span[i, 0]]
        return test_mask

if __name__=='__main__':
    device='cuda:0'
    model = Test().to(device)
    tensor_span = (torch.rand((1, 1), device=device)*100).int()
    print(tensor_span)
    output = model(tensor_span)
    print(output)
    traced_test = torch.jit.trace(model, tensor_span)
    print(traced_test.code)
    input_names = [ "test_input" ]
    output_names = [ "test_output" ]
    print('-------------------------------------')
    torch.onnx.export(model, 
                    tensor_span, 
                    "test.onnx", 
                    verbose=True,
                    opset_version=11,
                    input_names=input_names, 
                    output_names=output_names)