import torch
import numpy as np

pic = Image.open(...).convert('RGB')

img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))

img = torch.from_numpy(np.array(pic))