import torch
import torch.nn as nn

class SampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layers = torch.nn.Sequential(
            torch.nn.Conv2d(2, 2, kernel_size=1),
        )

        self.conv_layers = [
            torch.nn.Sequential(
                torch.nn.Conv2d(2, 2, kernel_size=1),
            )
        ]

    def forward(self, x):
        x = self.input_layers(x)
        for layer in self.conv_layers:
            x = layer(x)
        return x

model = SampleModel()

input_data = (torch.randn([2, 2, 1, 1]),)

model.eval()
graph_module = torch.export.export_for_training(model, input_data).module()

spec.target = param_buffer_table[spec.target]
KeyError: self___conv_layers_0_0.weight

class SampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = [
            torch.nn.Sequential(
                torch.nn.Conv2d(2, 2, kernel_size=1),
            )
        ]

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

class SampleModel(torch.nn.Module):
    def __init__(self, num_layers):
        super(SampleModel, self).__init__()
        self.num_layers = num_layers
        # Automatically create multiple convolutional layers
        for i in range(num_layers):
            conv_layer = torch.nn.Conv2d(2, 2, kernel_size=1)
            setattr(self, f'conv{i+1}', conv_layer)

    def forward(self, x):
        # Apply each conv layer to the input
        for i in range(1, self.num_layers + 1):
            layer = getattr(self, f'conv{i}')
            x = layer(x)
        return x

self.conv_layers = torch.nn.Sequential(torch.nn.Conv2d(2, 2, kernel_size=1))

self.conv_layers = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(2, 2, kernel_size=1))])