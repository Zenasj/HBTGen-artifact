py
import torch
import torch.nn.functional as F


# Resize the bigger side to the given size but keep aspect ratio
def resize(image: torch.Tensor, larger_size: int):
    image = image.unsqueeze(0)
    _, _, height, width = image.size()
    if height >= width:
        new_height = larger_size
        new_width = width * larger_size // height
    else:
        new_width = larger_size
        new_height = height * larger_size // width
    return F.interpolate(image, size=(new_height, new_width), mode="bilinear", align_corners=False).squeeze(0)


# Pad image to be divisible by a given value
def pad(image: torch.Tensor, divisible_by: int, value: float = 0.0):
    _, height, width = image.size()
    effective_height = int(torch.ceil(torch.tensor(height, dtype=torch.float) / divisible_by) * divisible_by)
    effective_width = int(torch.ceil(torch.tensor(width, dtype=torch.float) / divisible_by) * divisible_by)
    return F.pad(
        image,
        [0, effective_width - width, 0, effective_height - height],
        mode="constant",
        value=value,
    )

py
import torch
import torch.jit
import torch.nn as nn

from preprocess import pad, resize


# Define a preprocessing function that uses the building blocks of the preprocess module
# Renaming it, fixes the problem
def preprocess(image: torch.Tensor, size: int, divisible_by: int):
    image = resize(image, size)
    image = pad(image, divisible_by)
    return image


class Model(nn.Module):
    def __init__(self, size: int = 1024, divisible_by: int = 128):
        super(Model, self).__init__()
        self.size = size
        self.divisible_by = divisible_by

    def forward(self, image):
        image = preprocess(image, self.size, self.divisible_by)

        # Do something with the preprocessed image
        # ...


model = Model()

# Export the model
jit_model = torch.jit.script(model)
jit_model.save("model.pth")

# Load the model later
loaded_model = torch.jit.load("model.pth")