import argparse
import torch
import torch.utils.data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mwe')
    parser.add_argument('--num-workers', default=0, type=int)
    args = parser.parse_args()

    data = torch.rand(10, 1000)
    target = torch.rand(10)
    dataset = torch.utils.data.TensorDataset(data, target)
    data_loader = torch.utils.data.DataLoader(dataset,
        batch_size=2, num_workers=args.num_workers)
    for i, batch in enumerate(data_loader):
        pass