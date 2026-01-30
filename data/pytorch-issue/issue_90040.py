import torch.nn as nn

import os

import torch
from torch import nn


# a simple model class
class 测试模型类(nn.Module):
    def __init__(self):
        super().__init__()
        self.卷积1 = nn.Conv1d(1, 8, 2, stride=1, padding=1)

    def forward(self, x):
        return self.卷积1(x)


if __name__ == "__main__":
    # The path where the model is saved
    保存的路径 = "已训练的模型"
    if not os.path.isdir(保存的路径):
        os.makedirs(保存的路径)

    # generate a simple model
    某个测试模型 = 测试模型类()
    某个测试模型.cuda()
    某个测试模型.train()
    某个测试模型(torch.randn(10, 1, 1).cuda())

    临时文件路径字符串 = os.path.join(保存的路径, "一个简单的模型_1")
    # show error code
    torch.save(某个测试模型.state_dict(), 临时文件路径字符串)