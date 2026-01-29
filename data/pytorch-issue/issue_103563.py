# torch.rand(2, dtype=torch.float32)  # Inferred input shape for the model

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model_config = Configs({
            "a": {"aa": ["foo1", "foo12"], "ab": ["foo2", "foo22"]},
            "b": {"ba": ["foo3", "foo32"]},
            "c": {"ca": ["foo4"]}
        })

    def flatten_nested_dict(self, model_config) -> dict:
        assert(model_config.channels)
        embedding_sequence_groups = {}
        for (event_type, d) in model_config.channels.items():
            embedding_sequence_groups[event_type] = {}
            for entity_type, l in d.items():
                embedding_sequence_groups[event_type][entity_type] = l[0]
        return embedding_sequence_groups

    def wrapped_nested_dict(self, foo):
        return self.flatten_nested_dict(self.model_config)

    def forward(self, x):
        retval = self.wrapped_nested_dict(x)
        print(retval)
        return x + len(retval)

class Configs:
    def __init__(self, channels):
        self.channels = channels

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(2, dtype=torch.float32)

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)

