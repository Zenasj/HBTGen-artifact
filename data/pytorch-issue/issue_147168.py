import torch
import torch.nn as nn

def get_all_gather_streams(
        self, async_op: bool, training_state: TrainingState
    ) -> tuple[torch.Stream, torch.Stream]:
        if not async_op and training_state in (
            TrainingState.FORWARD,
            TrainingState.PRE_BACKWARD,
        ):
            # Use separate streams for implicit prefetching
            return self.all_gather_copy_in_stream, self.all_gather_stream
        
        # Use separate streams for explicit prefetching!
        current_stream = self.device_handle.current_stream()
        return current_stream, self.all_gather_stream # Change this!

class MLP(nn.Module):
    def __init__(self, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class MultiMLP(nn.Module):
    def __init__(self, hidden_dim: int, bias: bool = False, layers: int = 4):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_dim, bias=bias)
        self.mlps = nn.ModuleList([MLP(hidden_dim, bias) for _ in range(layers)])
        self.post_norm = nn.LayerNorm(hidden_dim, bias=bias)

    def forward(self, x):
        x = self.pre_norm(x)
        for mlp in self.mlps:
            x = x + mlp(x)
        x = self.post_norm(x)
        return x

class TestMemory(DTensorTestBase):
    @with_comms
    def test_over_allocation(self):
        mesh = init_device_mesh("cuda", (self.world_size,))
        device = torch.device("cuda")
        hidden_dim = 10240
        total_bsz = 16

        # ----- init model --------
        torch.manual_seed(0)
        model = MultiMLP(hidden_dim=hidden_dim).to(device).to(torch.float32)

        # --------  fsdp2 wrap --------
        fully_shard_fn = functools.partial(
            fully_shard,
            mesh=mesh,
            reshard_after_forward=True,
        )

        last_fsdp_module = None
        for module in model.modules():
            if isinstance(module, MLP):
                fully_shard_fn(module)
                if last_fsdp_module is not None:
                    last_fsdp_module.set_modules_to_forward_prefetch([module])
                    module.set_modules_to_backward_prefetch([last_fsdp_module])
                last_fsdp_module = module
        fsdp_model = fully_shard_fn(model)
        fsdp_model._set_unshard_async_op(True)

        optim = torch.optim.Adam(fsdp_model.parameters())

        # ----- init data -----
        torch.manual_seed(self.rank)
        bsz = total_bsz // self.world_size

        # --------  training loop --------
        torch.distributed.barrier()
        torch.cuda.synchronize(self.rank)
        
        train_iter = 4
        for iter in range(train_iter):
            # torch.distributed.barrier()
            # torch.cuda.synchronize(self.rank)

            if self.rank == 0 and iter == train_iter - 1:
                torch.cuda.memory._record_memory_history(max_entries=int(1E6))

            with record_function("## zero grad ##"):
                optim.zero_grad()

            input = torch.randn((bsz, hidden_dim), device="cuda")

            with record_function(f"## forward ##"):
                output = fsdp_model(input)
                loss = output.mean()

            with record_function(f"## backward ##"):
                loss.backward()

            with record_function("## optimizer step ##"):
                optim.step()

            if self.rank == 0 and iter == train_iter - 1:
                timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
                file_name = f"mem_{timestamp}"
                torch.cuda.memory._dump_snapshot(f"{file_name}.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)

        torch.distributed.barrier()
        torch.cuda.synchronize(self.rank)