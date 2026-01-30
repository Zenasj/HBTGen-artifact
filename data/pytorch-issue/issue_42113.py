import os
import time

import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

backend = 'qnnpack'
# backend = 'fbgemm'
import torch
torch.backends.quantized.engine = backend


class DownBlockQ(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.quant_input = QuantStub()
        self.dequant_output = DeQuantStub()

        self.conv1 = nn.Conv2d(in_ch, in_ch, 4, stride=2, padding=1, groups=in_ch)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # x = self.quant_input(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # x = self.dequant_output(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu1'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv2', 'bn2', 'relu2'], inplace=True)


class Model(nn.Module):
    def __init__(self, filters=22):
        super().__init__()
        self.quant_input = QuantStub()
        self.dequant_output = DeQuantStub()

        self.db1 = DownBlockQ(filters * 1, filters * 2)  # 128
        self.db2 = DownBlockQ(filters * 2, filters * 4)  # 64
        self.db3 = DownBlockQ(filters * 4, filters * 8)  # 32

    def forward(self, x):
        x = self.quant_input(x)
        x = self.db1(x)
        x = self.db2(x)
        x = self.db3(x)
        x = self.dequant_output(x)
        return x


def fuse_model(model):
    if hasattr(model, 'fuse_model'):
        model.fuse_model()

    for p in list(model.modules())[1:]:
        fuse_model(p)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def benchmark(func, iters=10, *args):
    t1 = time.time()
    for _ in range(iters):
        res = func(*args)
    print(f'{((time.time() - t1) / iters):.6f} sec')
    return res


def quantize():
    dummy = torch.rand(1, 22, 256, 256)
    # model = DownBlockQ(22 * 1, 22 * 2)
    model = Model(filters=22)
    model = model.eval()
    print("Before quantization")
    print_size_of_model(model)

    benchmark(model, 20, dummy)
    # print(model)
    fuse_model(model)

    model.qconfig = torch.quantization.get_default_qconfig(backend)
    # print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)

    # print(model)
    print("After quantization")
    print_size_of_model(model)
    benchmark(model, 20, dummy)
    # torch.jit.script(model).save('models/model_scripted.pt')


if __name__ == '__main__':
    quantize()

import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import torch.profiler
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import torch.quantization

data_path = '~/.data/imagenet'
saved_model_dir = 'data/'
# float_model_file = 'mobilenet_pretrained_float.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'
train_batch_size = 30
eval_batch_size = 50


def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file).to('cpu')
    model.eval()
    num_batches = 5
    with torch.profiler.profile(activities = [ProfilerActivity.CPU], record_shapes = True) as prof:
        with record_function("model_inference"):
            for i, (images, target) in enumerate(img_loader):
                if i < num_batches:
                    start = time.time()
                    output = model(images)
                    end = time.time()
                    elapsed = elapsed + (end-start)

                else:
                    break
        num_images = images.size()[0] * num_batches

        print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    print(prof.key_averages().table(sort_by = "cpu_time_total", row_limit = 10))
    return elapsed

def prepare_data_loaders(data_path):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(
           data_path, split="train",
        transform=transforms.Compose([
                   transforms.RandomResizedCrop(224),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   normalize,
               ]))
    dataset_test = torchvision.datasets.ImageNet(
          data_path, split="val",
              transform=transforms.Compose([
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  normalize,
              ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size, # pin_memory ="True",
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size, # pin_memory ="True",
        sampler=test_sampler)
    
    return data_loader, data_loader_test


data_loader, data_loader_test = prepare_data_loaders(data_path)
run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)
run_benchmark(saved_model_dir + scripted_quantized_model_file, data_loader_test)