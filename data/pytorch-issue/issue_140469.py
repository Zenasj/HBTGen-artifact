import torch

image_syn = torch.randn(size=(num_classes*args.dlipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.dldevice)
a = image_syn * 0.01
optimizer_img = torch.optim.SGD([image_syn, ], lr=args.dllr_img, momentum=0.5)