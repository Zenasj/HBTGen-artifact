import torch
import torch.nn as nn
import torch.nn.functional as F

class TestDummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Linear(8, 16)
        self.net2 = nn.Linear(16, 32)
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Linear(64, 8)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = F.relu(self.net4(x))
        return x

    def get_input(self):
        return torch.rand(8, 8, device="cuda")


class FsdpTpSaveLoadTest(DTensorTestBase):
    @property
    def world_size(self):
        gpu_num = torch.cuda.device_count()
        return gpu_num if gpu_num % 2 == 0 and gpu_num > 4 else 4

    @with_comms
    def test_save_load(self):
        dummy_model = TestDummyModel().cuda()
        mesh_2d = init_device_mesh(
            self.device_type,
            (2, self.world_size // 2),
            mesh_dim_names=("dp", "tp"),
        )
        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model = parallelize_module(dummy_model, tp_mesh, parallelize_plan)
        model = FSDP(model, device_mesh=dp_mesh, use_orig_params=False)

        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(model, options=options)
        print(f'!!! rank {self.rank}: {len(state_dict) = }', flush=True)
        set_model_state_dict(model, state_dict, options=options)