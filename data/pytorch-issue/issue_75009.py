def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image


model = models.shufflenet_v2_x1_0(pretrained=True)
model.eval()


img_file = 'ILSVRC2012_val_00000001.JPEG'

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

img = Image.open(img_file)

x = transform(img).unsqueeze(0)
x = torch.cat([x, x.clone()], dim=0)

model.cuda()
y = model(x.cuda())
pred = y.argmax(1)
print(pred)

def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)

            min_v, max_v = torch._aminmax(x2)
            scale = (max_v - min_v) / 255
            zero_point = torch.round((0 - min_v) / scale).to(torch.int)

            x2 = torch.fake_quantize_per_tensor_affine(
                x2, scale.item(), zero_point.item(), 0, 255)
            
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)

            # min_v, max_v = torch.aminmax(x2)
            min_v, max_v = torch.tensor(0.0), torch.tensor(2.6)
            scale = (max_v - min_v) / 255
            zero_point = torch.round((0 - min_v) / scale).to(torch.int)

            x2 = torch.fake_quantize_per_tensor_affine(
                x2, scale.item(), zero_point.item(), 0, 255)
            
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out