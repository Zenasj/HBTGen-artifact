import torch.nn as nn

import os
import types

import pytest
import torch
from torch.multiprocessing import Process


def init_processes(x, conv_weight_distributed, rank):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = '2'
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl')
    channels = x.size(1)
    conv_distributed = torch.nn.parallel.DistributedDataParallel(
            torch.nn.Conv2d(channels, channels, 1).cuda(),
            device_ids=[rank])
    conv_distributed.module.weight = torch.nn.Parameter(
            conv_weight_distributed)
    torch.nn.init.constant_(conv_distributed.module.bias, 0)
    y = conv_distributed(x)
    y.mean().backward()
    # it to the dict manually
    state_dict = {
            'x': x,
            'x_grad': x.grad,
            'conv': conv_distributed.module,
            'conv_weight_grad': conv_distributed.module.weight.grad,
            'conv_bias_grad': conv_distributed.module.bias.grad,
            'y': y,
    }
    torch.save(state_dict, '{}.pth'.format(rank))


@pytest.fixture(scope='module')
def conv_instances():
    torch.multiprocessing.set_start_method('spawn')

    x_local = torch.randn(
            (4, 2, 1, 1), device='cuda:0', requires_grad=True)
    x_distributed = x_local.detach().cuda().requires_grad_(True)

    channels = x_local.size(1)
    conv_local = torch.nn.Conv2d(channels, channels, 1).cuda()
    torch.nn.init.constant_(conv_local.bias, 0)
    y_local = conv_local(x_local)
    y_local.mean().backward()

    WORLD_SIZE = 2
    step = x_local.size(0) // WORLD_SIZE
    x_0 = x_distributed.detach(
            )[:step, :, :, :].clone().cuda(0).requires_grad_(True)
    x_1 = x_distributed.detach(
            )[step:, :, :, :].clone().cuda(1).requires_grad_(True)
    conv_weight_distributed_0 = conv_local.weight.detach(
            ).clone().cuda(0)
    conv_weight_distributed_1 = conv_local.weight.detach(
            ).clone().cuda(1)
    process_1 = Process(
            target=init_processes,
            args=(x_0, conv_weight_distributed_0, 0))
    process_2 = Process(
            target=init_processes,
            args=(x_1, conv_weight_distributed_1, 1))
    process_1.start()
    process_2.start()

    process_1.join()
    process_2.join()
    conv_instance_local = types.SimpleNamespace()
    conv_instance_local.x = x_local
    conv_instance_local.conv = conv_local
    conv_instance_local.y = y_local
    conv_instance_distributed_0 = types.SimpleNamespace(
            **torch.load('0.pth'))
    conv_instance_distributed_1 = types.SimpleNamespace(
            **torch.load('1.pth', map_location={'cuda:1': 'cuda:0'}))
    yield (
            conv_instance_local,
            conv_instance_distributed_0,
            conv_instance_distributed_1,
    )
    os.remove('0.pth')
    os.remove('1.pth')


def allclose(x, y):
    adiff = (x - y).abs()
    if (y == 0).all():
        rdiff = 'NaN'
    else:
        rdiff = (adiff / y).abs().max()
    assert torch.allclose(x, y), (
            'Tensor close check failed\n'
            '{}\n'
            '{}\n'
            'adiff={}\n'
            'rdiff={}\n'
    ).format(x, y, adiff, rdiff)


def test_x(conv_instances):
    local, distributed_0, distributed_1 = conv_instances
    x_distributed = torch.cat(
            (distributed_0.x, distributed_1.x))
    allclose(local.x, x_distributed)


def test_parameters(conv_instances):
    local, distributed_0, distributed_1 = conv_instances
    a = list(local.conv.parameters())
    b = list(distributed_0.conv.parameters())
    c = list(distributed_0.conv.parameters())
    for x, y in zip(a, b):
        allclose(x, y)
    for x, y in zip(a, c):
        allclose(x, y)


def test_conv_weight_grad_between_local_and_distributed(
        conv_instances):
    local, distributed_0, distributed_1 = conv_instances
    allclose(
            local.conv.weight.grad,
            distributed_0.conv_weight_grad)


def test_conv_weight_grad_between_distributeds(
        conv_instances):
    local, distributed_0, distributed_1 = conv_instances
    allclose(
            distributed_0.conv_weight_grad,
            distributed_1.conv_weight_grad)


def test_conv_bias_grad_between_local_and_distributed(
        conv_instances):
    local, distributed_0, distributed_1 = conv_instances
    allclose(
            local.conv.bias.grad,
            distributed_0.conv_bias_grad)


def test_conv_bias_grad_between_distributeds(
        conv_instances):
    local, distributed_0, distributed_1 = conv_instances
    allclose(
            distributed_0.conv_bias_grad,
            distributed_1.conv_bias_grad)


def test_y(conv_instances):
    local, distributed_0, distributed_1 = conv_instances
    y_distributed = torch.cat((distributed_0.y, distributed_1.y))
    allclose(local.y, y_distributed)