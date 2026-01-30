import torch
from torchvision import transforms

height = 3
width = 4
resize = 2
tensor3 = torch.rand(3,height,width)
tensor1 = torch.zeros(1,height,width)
#tensor1 = torch.rand(1,height,width)

imageToTensor = transforms.ToTensor()
tensorToImage = transforms.ToPILImage()

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(resize, scale=(1.0, 1.0)),
    transforms.ToTensor(),
])

tensor4 = torch.cat((tensor3,tensor1),0)

image4 = tensorToImage(tensor4)
transformed_image4 = train_transform(image4)

print(tensor4)
print(transformed_image4)