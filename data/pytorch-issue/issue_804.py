import torch

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder("data/everything/", transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])),
        batch_size=16, shuffle=False,
        num_workers=3, pin_memory=False)