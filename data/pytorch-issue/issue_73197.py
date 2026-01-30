import torchvision

model = torchvision.models.resnet18(num_classes=1000)
model.to(device="lazy")
model.train()
copy.deepcopy(model) # error