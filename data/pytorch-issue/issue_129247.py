import torch.nn as nn

import torch

class InputContainer(torch.nn.Module):
    def __init__(self, obj):
        super().__init__()
        for key, value in obj.items():
            setattr(self, key, value)

inputs = {
        "audio" : torch.rand((553172,))
        }
inputs_model = InputContainer(inputs)
inputs_script = torch.jit.script(inputs_model)
inputs_script.save("rmvpe_inputs.pt")
inputs_script.get_parameter("audio")