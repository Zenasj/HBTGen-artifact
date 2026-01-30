import torch

train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

train_dataloader = DataLoader(dataset,
                              batch_size=args.batch_size,
                              pin_memory=False,
                              num_workers=args.num_workers,
                              sampler=train_subsampler)
val_dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            pin_memory=False,
                            num_workers=args.num_workers,
                            sampler=val_subsampler)
dataloaders = {'train': train_dataloader, 'val': val_dataloader}

model = Net(model_name=args.model,
                    num_classes=args.num_classes,
                    num_channels=args.num_channels,
                    temporal_size=args.temporal_size)

model.to(args.device)
swa_model = AveragedModel(model, device=args.device)