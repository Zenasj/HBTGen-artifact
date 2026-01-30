import torch.nn as nn

import torch
import matplotlib.pyplot as plt

img = plt.imread("lena.jpeg")
img = img[110:120, 110:120]
img = img.reshape(1,10,10,3)
img_tensor = torch.from_numpy(img)
img_tensor = img_tensor.permute(0, 3, 1, 2)

theta = torch.tensor([[1, 0, 0], [0, 1, 0]])
theta = theta.view(1, 2, 3)

grid = torch.nn.functional.affine_grid(theta, img_tensor.size())
out = torch.nn.functional.grid_sample(img_tensor.float(), grid.float())

out_img = out.permute(0, 2, 3, 1).data.numpy()
out_img = out_img.reshape(10, 10, 3)
plt.imshow(out_img.astype(int))
plt.show()