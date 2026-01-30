import torch

train_dataset = lmdbDataset(root=opt.trainroot, transform=resizeNormalize(size=(592, 32)))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=None,
    num_workers=int(opt.workers),
    collate_fn=alignCollate())
train_iter = iter(train_loader)
cpu_images, cpu_texts, cpu_lengths = next(train_iter)