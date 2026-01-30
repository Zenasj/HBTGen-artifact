import torch

# setup model & dataloaders
steps_per_epoch = len(train_dataloader) 
epochs = epochs# any number
# add optimizer
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                      max_lr=1e-1,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs, verbose=True)
#setup trainer & fit