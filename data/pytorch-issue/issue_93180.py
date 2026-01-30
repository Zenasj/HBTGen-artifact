import torch
import torch._dynamo as dynamo

from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])])

def f(img):
    return transform(img)

im = Image.new(mode="RGB", size=(20, 20))

opt_f = dynamo.optimize("inductor")(f)
print(opt_f(im))

def is_safe_constant(v):
    if istype(v, (tuple, frozenset)):
        return all(map(is_safe_constant, v))
    return istype(
        v,
        (
            types.CodeType,
            int,
            float,
            bool,
            str,
            bytes,
            type(None),
            slice,
            type(type),
            torch.device,
        ),
    ) or isinstance(v, enum.Enum)