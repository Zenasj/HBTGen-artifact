import torch
model = torch.jit.load("mobilenet_v2_imagenet.pt").cuda()
model.eval()
rand_input = torch.rand((1,3,224,224)).cuda()
model(rand)