import torch

generator = torch.Generator()
generator.manual_seed(2022)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=1, generator=generator)
# Don't pass in the same generator to `valid_loader` if you don't want the RNG to share states and influence one another.