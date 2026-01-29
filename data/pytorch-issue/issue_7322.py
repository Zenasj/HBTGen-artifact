# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: (batch_size, 3, 32, 32)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1_0 = self.Conv_Module(3, 6, 1, 3, 1)
        self.layer1_1 = self.Conv_Module(1, 6, 4, 3, 1)
        self.Dense = self.Classification(384)

    def Conv_Module(self, in_filters, out_filters, stride, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_filters, out_channels=out_filters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU())

    def Classification(self, num_in):
        return nn.Sequential(
            nn.Linear(num_in, 10))

    def forward(self, original_img):
        first_conv = self.layer1_0(original_img)
        attention_map = first_conv.mean(1).unsqueeze(dim=1)
        inverted_attention = -attention_map
        output_mask = torch.ge(attention_map, inverted_attention).float()
        second_conv = self.layer1_1(output_mask)
        classifier_input = second_conv.view(-1, 384)
        classifier_output = self.Dense(classifier_input)
        return F.log_softmax(classifier_output, dim=1)

def my_model_function():
    return MyModel()

def GetInput():
    # CIFAR-10 images are 32x32 with 3 channels
    batch_size = 128
    return torch.rand(batch_size, 3, 32, 32, dtype=torch.float32)

