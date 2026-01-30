import torch

def main():
    torch.set_num_threads(4)

    dataloader = torch.utils.data.DataLoader([1, 2, 3], num_workers=1)
    iter(dataloader).next()

    return

if __name__ == '__main__':
    main()