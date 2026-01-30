import torch
import torch.nn as nn
import torch.jit
import torch.onnx

@torch.jit.script
def check_init(input_data, hidden_size, prev_state):
    # type: (torch.Tensor, int, torch.Tensor) -> torch.Tensor
    batch_size = input_data.size(0)
    spatial_size_0 = input_data.size(2)
    spatial_size_1 = input_data.size(3)
    # generate empty prev_state, if None is provided
    state_size = (2, batch_size, hidden_size ,spatial_size_0, spatial_size_1)
    state = torch.zeros(state_size, device=input_data.device)
    if prev_state.size(0) != 0:
        state[:] = prev_state
    return state

class Example(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_data, prev_state):
        prev_state = check_init(input_data, self.hidden_size, prev_state)
        return prev_state

random_data = torch.rand((1,5,30,30))
empty_tensor = torch.tensor([], dtype=torch.float).view(0,0,0,0,0)
traced_module = torch.jit.trace(Example(10), (random_data, empty_tensor,))
outputs = list(traced_module(random_data, empty_tensor))

torch.onnx.export(traced_module,
                  (random_data, empty_tensor),
                  "test.onnx",
                  example_outputs=outputs,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input_data', 'prev_state'],
                  output_names=['state'],
                  dynamic_axes={
                                'prev_state': {0: 'state_num', 1: 'batch_size', 2:'hidden_dim', 3: 'height', 4: 'width'},
                                'state': {1: 'batch_size', 3: 'height', 4: 'width'},
                              }, verbose=True)