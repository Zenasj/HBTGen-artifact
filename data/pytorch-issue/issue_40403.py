import random
import torch
import torch.utils.data as data


class DummyDataset(data.Dataset):
    def __init__(self, num_classes):
        super(DummyDataset, self).__init__()
        # Create tensor on GPU 
        self.tensor = torch.randn(3, 224, 224, device=torch.device('cuda'))
        self.num_classes = num_classes

    def __getitem__(self, index):
        torch.manual_seed(index)
        random.seed(index)
        return self.tensor, \
               random.randint(0, self.num_classes - 1)

import random
import torch
torch.multiprocessing.set_start_method('spawn')# good solution !!!!
import torch.utils.data as data


class DummyDataset(data.Dataset):
    def __init__(self, num_classes):
        super(DummyDataset, self).__init__()
        # Create tensor on GPU 
        self.tensor = torch.randn(3, 224, 224, device=torch.device('cuda'))
        self.num_classes = num_classes

    def __getitem__(self, index):
        time.sleep(0.1);# !!!!!! In order to test, should be have virtual process time !!!!!!
        torch.manual_seed(index)
        random.seed(index)
        return self.tensor, \
               random.randint(0, self.num_classes - 1)

import random
import torch
import torch.utils.data as data

class DummyDataset(data.Dataset):
    def __init__(self, num_classes):
        super(DummyDataset, self).__init__()
        # Create tensor on GPU 
        self.tensor = torch.randn(3, 224, 224, device=torch.device('cuda'))
        self.num_classes = num_classes

    def __getitem__(self, index):
        torch.manual_seed(index)
        random.seed(index)
        return self.tensor, \
               random.randint(0, self.num_classes - 1)

torch.multiprocessing.set_start_method('spawn')# good solution !!!!

if __name__=="__main__":
    args = parse_args()

    torch.cuda.set_device(args.gpu)

    # Initialize distributed communication
    torch.distributed.init_process_group(backend='nccl',
                                         init_method="env://",
                                         world_size=world_size)


    # Define bunch of variables

    dataset = DummyDataset(num_classes=num_classes)
    model = ImageNetClassifier(
        num_classes=num_classes,
        ignore_label=ignore_label,
        label_smoothing=label_smoothing,
        base_model="resnet50"
    )
    print('Finish instantiating model')
    
    # Run training loop below

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    main()

DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0)

py
# other imports elided
from typing import Tuple

from torch import Tensor
from torch.util.data import Dataset

class MyDataset(Dataset):
    # elided
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        features = prep_feature(self.raw_data[index])
        features = torch.tensor(features, device=self.device)
        targets = torch.tensor(fetch_targets(index), device=self.device) # <--- throws the error
        return features, targets

py
torch.multiprocessing.set_start_method('spawn')