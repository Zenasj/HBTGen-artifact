import torch
import torch.nn as nn

print(torch.__version__)

class test(torch.nn.Module):

    def __init__(self, vocab_size=10, rnn_dims=512):
        super().__init__()

    def forward(self, x):
        # CASE 1) RandomNormalLike - Works
        # mask = torch.randn_like(x).to(torch.float32)

        # CASE 2) RandomUniformLike - Broken
        # mask = torch.rand_like(x).to(torch.float32)

        # CASE 3) RandomNormal - Broken
        # mask = torch.randn(x.size()).to(torch.float32)

        # CASE 4) RandomUniform - Broken
        mask = torch.rand(x.size()).to(torch.float32)

        return mask


# PyTorch model
model = test()

input = torch.ones((1,256)).to(torch.float32).cuda()
output = model(input)
torch.onnx.export(model,
                  input,
                  'test_rand.onnx',
                  example_outputs=output)