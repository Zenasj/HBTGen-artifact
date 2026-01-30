import sys
import os

def _preload_cuda_deps():
    """Preloads cudnn/cublas deps if they could not be found otherwise."""
    # Should only be called on Linux if default path resolution have failed

    cuda_libs = {
        'cublas': 'libcublas.so.11',
        'cudnn': 'libcudnn.so.8',
        'cuda_nvrtc': 'libnvrtc.so.11.2',
        'cuda_runtime': 'libcudart.so.11.0',
        'cuda_cupti': 'libcupti.so.11.7',
        'cufft': 'libcufft.so.10',
        'curand': 'libcurand.so.10',
        'cusolver': 'libcusolver.so.11',
        'cusparse': 'libcusparse.so.11',
        'nccl': 'libnccl.so.2',
        'nvtx': 'libnvToolsExt.so.1',
    }
    cuda_libs_paths = {lib_folder: None for lib_folder in cuda_libs.keys()}

    for path in sys.path:
        nvidia_path = os.path.join(path, 'nvidia')
        if not os.path.exists(nvidia_path):
            continue
        for lib_folder, lib_name in cuda_libs.items():
            candidate_path = os.path.join(nvidia_path, lib_folder, 'lib', lib_name)
            if os.path.exists(candidate_path) and not cuda_libs_paths[lib_folder]:
                cuda_libs_paths[lib_folder] = candidate_path
        if all(cuda_libs_paths.values()):
            break
    if not all(cuda_libs_paths.values()):
        none_libs = [lib for lib in cuda_libs_paths if not cuda_libs_paths[lib]]
        raise ValueError(f"{', '.join(none_libs)} not found in the system path {sys.path}")


_preload_cuda_deps()

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_id)
    loss_fn(outputs, labels).backward()
    optimizer.step()

if __name__ == "__main__":
    demo_basic()