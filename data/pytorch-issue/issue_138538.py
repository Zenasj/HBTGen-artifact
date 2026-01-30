import torch.nn as nn

def forward(self, inputs, sizes, hooks):
    # No stacktrace found for following nodes
    getitem: "f32[]" = inputs[0]
    getitem_1: "f32[64, 128]" = inputs[1]
    getitem_2: "f32[64, 128]" = inputs[2]
    getitem_3: "f32[64, 128]" = inputs[3]
    getitem_4: "f32[64, 128]" = inputs[4]
    getitem_5: "f32[128, 128]" = inputs[5]
    getitem_6: "f32[128, 128]" = inputs[6]
    getitem_7: "f32[128, 128]" = inputs[7]
    getitem_8: "f32[128, 128]" = inputs[8]
    getitem_9: "f32[128, 128]" = inputs[9]
    getitem_10: "f32[128, 128]" = inputs[10]
    getitem_11: "f32[128, 128]" = inputs[11];  inputs = None
    expand: "f32[64, 128]" = torch.ops.aten.expand.default(getitem, [64, 128]);  getitem = None
    permute: "f32[128, 64]" = torch.ops.aten.permute.default(expand, [1, 0])
    mm: "f32[128, 128]" = torch.ops.aten.mm.default(permute, getitem_4);  permute = getitem_4 = None
    permute_1: "f32[128, 128]" = torch.ops.aten.permute.default(mm, [1, 0]);  mm = None
    mm_1: "f32[64, 128]" = torch.ops.aten.mm.default(expand, getitem_5);  expand = getitem_5 = None
    permute_2: "f32[128, 128]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    accumulate_grad_ = torch.ops.inductor.accumulate_grad_.default(getitem_8, permute_2);  getitem_8 = accumulate_grad_ = None
    permute_3: "f32[128, 64]" = torch.ops.aten.permute.default(mm_1, [1, 0])
    mm_2: "f32[128, 128]" = torch.ops.aten.mm.default(permute_3, getitem_3);  permute_3 = getitem_3 = None
    permute_4: "f32[128, 128]" = torch.ops.aten.permute.default(mm_2, [1, 0]);  mm_2 = None
    mm_3: "f32[64, 128]" = torch.ops.aten.mm.default(mm_1, getitem_6);  mm_1 = getitem_6 = None
    permute_5: "f32[128, 128]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    accumulate_grad__1 = torch.ops.inductor.accumulate_grad_.default(getitem_9, permute_5);  getitem_9 = accumulate_grad__1 = None
    permute_6: "f32[128, 64]" = torch.ops.aten.permute.default(mm_3, [1, 0])
    mm_4: "f32[128, 128]" = torch.ops.aten.mm.default(permute_6, getitem_2);  permute_6 = getitem_2 = None
    permute_7: "f32[128, 128]" = torch.ops.aten.permute.default(mm_4, [1, 0]);  mm_4 = None
    mm_5: "f32[64, 128]" = torch.ops.aten.mm.default(mm_3, getitem_7);  mm_3 = getitem_7 = None
    permute_8: "f32[128, 128]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    accumulate_grad__2 = torch.ops.inductor.accumulate_grad_.default(getitem_10, permute_8);  getitem_10 = accumulate_grad__2 = None
    permute_9: "f32[128, 64]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    mm_6: "f32[128, 128]" = torch.ops.aten.mm.default(permute_9, getitem_1);  permute_9 = getitem_1 = None
    permute_10: "f32[128, 128]" = torch.ops.aten.permute.default(mm_6, [1, 0]);  mm_6 = None
    permute_11: "f32[128, 128]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    accumulate_grad__3 = torch.ops.inductor.accumulate_grad_.default(getitem_11, permute_11);  getitem_11 = accumulate_grad__3 = None
    getitem_12 = hooks[0]
    call_hook = torch__dynamo_external_utils_call_hook(getitem_12, [], [permute_2]);  getitem_12 = permute_2 = call_hook = None
    getitem_13 = hooks[1]
    call_hook_1 = torch__dynamo_external_utils_call_hook(getitem_13, [], [permute_5]);  getitem_13 = permute_5 = call_hook_1 = None
    getitem_14 = hooks[2]
    call_hook_2 = torch__dynamo_external_utils_call_hook(getitem_14, [], [permute_8]);  getitem_14 = permute_8 = call_hook_2 = None
    getitem_15 = hooks[3];  hooks = None
    call_hook_3 = torch__dynamo_external_utils_call_hook(getitem_15, [], [permute_11]);  getitem_15 = permute_11 = call_hook_3 = None
    return []

import torch
import torch.distributed as dist
from torch._dynamo import compiled_autograd
import os


world_size = 1


def print_rank0(str):
    if dist.get_rank() == 0:
        print(str)


def grad_sync(param):
    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)


class Module(torch.nn.Module):
    def __init__(self, ioc):
        super().__init__()
        self.fc1 = torch.nn.Linear(ioc, ioc, bias=False)
        self.fc2 = torch.nn.Linear(ioc, ioc, bias=False)
        self.fc3 = torch.nn.Linear(ioc, ioc, bias=False)
        self.fc4 = torch.nn.Linear(ioc, ioc, bias=False)

        self.grad_acc_hooks = []
        self.grad_acc = []
        self.params = [self.fc1.weight, self.fc2.weight,
                       self.fc3.weight, self.fc4.weight]
        for i, param in enumerate(self.params):

            def wrapper(param):
                param_tmp = param.expand_as(param)
                grad_acc = param_tmp.grad_fn.next_functions[0][0]

                def grad_acc_hook(*notneeded):
                    grad_sync(param)

                self.grad_acc.append(grad_acc)
                self.grad_acc_hooks.append(
                    grad_acc.register_hook(grad_acc_hook))

            wrapper(param)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x.sum()


def compiler_fn(gm):
    return torch.compile(gm, backend="inductor", fullgraph=True)


def run(rank):
    bs = 64
    ioc = 128

    model = Module(ioc)
    model_to_train = torch.compile(model, backend="inductor")

    input = torch.randn([bs, ioc])
    loss = model_to_train(input)
    with compiled_autograd.enable(compiler_fn):
        loss.backward()
    print(f"finished on rank {rank}")


def init_process(size, rank, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank)


if __name__ == "__main__":
    init_process(world_size, 0, run)