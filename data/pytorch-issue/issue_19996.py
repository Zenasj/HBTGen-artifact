import torch

dataloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=arg.batch_size, shuffle=True,
                                                   num_workers=arg.num_workers, pin_memory=True)

dataloader_2 = torch.utils.data.DataLoader(trainset_2, batch_size=arg.batch_size, shuffle=True,
                                                   num_workers=arg.num_workers, pin_memory=True)

dataloader_3 = torch.utils.data.DataLoader(trainset_3, batch_size=arg.batch_size, shuffle=True,
                                                   num_workers=arg.num_workers, pin_memory=True)