from PIL import Image
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torch
 

img1 = Image.open('00035.png')
img2 = Image.open('00036.png')

transform = transforms.Compose([
    transforms.PILToTensor()
])


t_img1 = transform(img1).to(torch.device("mps"), dtype=torch.float32)  / 256.0
t_img2 = transform(img2).to(torch.device("mps"), dtype=torch.float32)  / 256.0

grid = make_grid([t_img1, t_img2], nrow=1)

gridImage =  transforms.ToPILImage()(grid.cpu());

gridImage.save('mps_grid.png')

t_img1 = transform(img1).to(torch.device("cpu"), dtype=torch.float32)  / 256.0
t_img2 = transform(img2).to(torch.device("cpu"), dtype=torch.float32)  / 256.0

grid = make_grid([t_img1, t_img2], nrow=1)

gridImage =  transforms.ToPILImage()(grid.cpu());

gridImage.save('cpu_grid.png')

from PIL import Image
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torch


img1 = Image.open('00035.png')
img2 = Image.open('00036.png')

transform = transforms.Compose([
    transforms.PILToTensor()
])


t_img1 = transform(img1).to(torch.device("mps"), dtype=torch.float32)  / 256.0
t_img2 = transform(img2).to(torch.device("mps"), dtype=torch.float32)  / 256.0

grid = make_grid([t_img1, t_img2], nrow=1)

gridImageMPS =  transforms.ToPILImage()(grid.cpu());

gridImageMPS.save('mps_grid.png')


t_img1 = transform(img1).to(torch.device("cpu"), dtype=torch.float32)  / 256.0
t_img2 = transform(img2).to(torch.device("cpu"), dtype=torch.float32)  / 256.0

grid = make_grid([t_img1, t_img2], nrow=1)

gridImage =  transforms.ToPILImage()(grid.cpu());

gridImage.save('cpu_grid.png')

assert  list(gridImageMPS.getdata()) == list(gridImage.getdata()), 'the images  are different'