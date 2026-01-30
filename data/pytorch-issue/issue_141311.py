import torch.nn as nn
import torchvision

3
import torch
from torchvision.transforms.functional import resize


class ImgPreprocessor(torch.nn.Module):
    def __init__(self):
        super(ImgPreprocessor, self).__init__()
        self.patch_size = 14


    def reshape_by_patch(self, image):
        # CHW
        patch_size = self.patch_size
        patches = torch.nn.functional.unfold(
            image,
            (patch_size, patch_size),
            stride=(patch_size, patch_size)
        )

        patches = patches.reshape(image.size(0), patch_size, patch_size, -1)
        patches = patches.permute(0, 1, 3, 2).reshape(image.size(0), patch_size, -1)
        return patches

    def forward(self, image: torch.Tensor) -> torch.FloatTensor:
        # H * W * C
        # width, height = 1920, 1080
        # C H W
        image = image.permute(2, 0, 1)
        refine_img = resize(image, [952, 1680])
        mean = torch.FloatTensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1)
        std = torch.FloatTensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1)

        patches = []
        width, height = refine_img.shape[2], refine_img.shape[1]
        grid_x = int(width / 4)
        grid_y = int(height / 2)
        for i in range(0, height, grid_y):
            rows = []
            for j in range(0, width, grid_x):
                rows.append(refine_img[:,i:i + grid_y, j:j+grid_x])
            patches.append(rows)

        thumb_img = resize(image, [336, 602])
        slice_imgs = [thumb_img]
        slice_imgs.extend(patches[0])
        slice_imgs.extend(patches[1])

        image_patches = [img.float()/255 for img in slice_imgs]
        image_patches = [img.sub(mean).div(std) for img in image_patches]
        image_patches = [self.reshape_by_patch(img) for img in image_patches]
        pixel_values = [i.flatten(end_dim=1).permute(1, 0) for i in image_patches]
        pixel_values = torch.nn.utils.rnn.pad_sequence(pixel_values, batch_first=True, padding_value=0.0)
        pixel_values = pixel_values.permute(0, 2, 1).reshape(9, 3, -1, 14448)
        return pixel_values

# ImgPreprocessor().forward(torch.ones([1080, 1920, 3]))


from executorch.exir import to_edge
# 1. torch.export: Defines the program with the ATen operator set.
aten_dialect = torch.export.export(ImgPreprocessor(), (torch.ones([1080, 1920, 3]),))

# 2. to_edge: Make optimizations for Edge devices
edge_program = to_edge(aten_dialect)

# 3. to_executorch: Convert the graph to an ExecuTorch program
executorch_program = edge_program.to_executorch()

# 4. Save the compiled .pte program
with open("minicpmv_preprocessor.pte", "wb") as file:
    file.write(executorch_program.buffer)