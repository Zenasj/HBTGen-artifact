import torch
from torchvision import models
import torch._dynamo as dynamo
dynamo.config.verbose=True
dynamo.reset()

r18 = models.resnet18().to("cuda").eval()

trt_compiled_r18 = dynamo.optimize("tensorrt")(r18)

input_image_pytorch = torch.randn((1, 3, 480, 768), requires_grad=False).to("cuda").detach()

trt_compiled_r18(input_image_pytorch)