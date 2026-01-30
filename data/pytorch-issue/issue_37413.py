import torch

indices = torch.arange(len(dataset)).tolist()
sub_dataset = torch.utils.data.Subset(dataset_test, indices[:1])
data_loader_test = torch.utils.data.DataLoader(sub_dataset, batch_size=1, shuffle=False, num_workers=num_workers,collate_fn=utils.collate_fn)
evaluate(model, data_loader_test, device=device)

indices = torch.randperm(len(dataset)).tolist()