import torch
from mlp_mixer_pytorch import MLPMixer
import torchdynamo

model = MLPMixer(
    image_size = 256,
    channels = 3,
    patch_size = 16,
    dim = 512,
    depth = 12,
    num_classes = 1000
).cuda()

img = torch.randn(1, 3, 256, 256).cuda()

@torchdynamo.optimize("inductor")
def pred():
    pred = model(img) # (1, 1000)

pred()