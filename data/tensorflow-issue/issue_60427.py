py
import ai_edge_torch
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding='valid')
        self.dense1 = nn.Linear(32, 6)
        self.dense2 = nn.Linear(6*(input_shape[1]-2)*(input_shape[2]-2), num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = torch.permute(x, (0, 2, 3, 1)) # Dense is operating on the channels dimension
        x = self.dense1(x)
        x = x.view(x.size(0), -1)
        x = self.dense2(x)
        return self.softmax(x)


image_shape = (3, 32, 32)
model = SimpleModel(image_shape, 10)
sample_input = (torch.randn(1, 3, 32, 32),)

edge_model = ai_edge_torch.convert(model.eval(), sample_input)
edge_model.export("test.tflite")