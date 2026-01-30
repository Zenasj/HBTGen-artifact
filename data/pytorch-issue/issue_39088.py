3

import torch
import torchvision

# Windows requires ``if __name__ == '__main__':`` for worker processes.
if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(
        root='./', transform=torchvision.transforms.ToTensor(), download=True)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)
    #------------------------------------------------------#
    # 1st stage
    print('cpu default')
    torch.set_default_tensor_type(torch.FloatTensor)
    try:
        for data in dataloader:
            print(data[0].device)
            break
    except Exception as e:
        print(e)
    #------------------------------------------------------#
    # 2nd stage
    print('move to gpu')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    try:
        for data in dataloader:
            print(data[0].device)
            break
    except Exception as e:
        print(e)
    #------------------------------------------------------#
    # 3rd stage
    print('move to cpu')
    torch.set_default_tensor_type(torch.FloatTensor)
    try:
        for data in dataloader:
            print(data[0].device)
            break
    except Exception as e:
        print(e)

3 
import torch
import torchvision

torch.set_default_tensor_type(torch.cuda.FloatTensor)

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    dataset = torchvision.datasets.CIFAR10(
        root='./', transform=torchvision.transforms.ToTensor(), download=True)

    #------------------------------------------------------#
    # 1st stage
    print()
    print('1st stage')
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=0)
    try:
        for data in dataset:
            print(data[0].device)
            print(type(data[1]))
            print()
            break
    except Exception as e:
        print(e)
    #------------------------------------------------------#
    # 2nd stage
    print()
    print('2nd stage')
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=0)
    try:
        for data in dataloader:
            print(data[0].device)
            print(data[1].device)
            print()
            break
    except Exception as e:
        print(e)