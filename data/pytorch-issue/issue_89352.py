import torch

if __name__ == "__main__":
    print(torch.__version__)
    dl = torch.utils.data.DataLoader(dataset=dts(), num_workers=16, batch_size=2)
    for data in dl:
        debug = "debug"
        print(data.shape)