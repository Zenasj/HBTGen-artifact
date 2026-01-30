import torch
import random

class MyColorJitter(transforms.ColorJitter):
  def init(self, brightness, contrast, saturation, hue, seed):
    super().init(brightness, contrast, saturation, hue)
    self.seed = seed
  
  def forward(self, img):
    torch.manual_seed(self.seed)
    random.seed(self.seed)
    return super().forward(img)

color_jitter = MyColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, seed=seed)