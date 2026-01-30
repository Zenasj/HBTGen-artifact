import torch.nn.functional as F

import os
import torch.distributed
import torch.nn as nn
from torch.distributed._tensor.api import Replicate, Shard, DTensor
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("tp",))
torch.cuda.set_device(int(os.getenv("LOCAL_RANK")))


class Embedding_TP(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__(num_embeddings // torch.distributed.get_world_size(), embedding_dim)
        self.weight = nn.Parameter(DTensor.from_local(self.weight, device_mesh=mesh, placements=[Shard(0)]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = DTensor.from_local(input, device_mesh=mesh, placements=[Replicate()])
        input = super().forward(input)
        input = input.redistribute(placements=[Replicate()])
        return input.to_local()
    
linear = nn.Linear(23, 29).to(torch.cuda.current_device())
l1 = Embedding_TP(100, 23).to(torch.cuda.current_device())
l2 = Embedding_TP(101, 29).to(torch.cuda.current_device())

x = torch.tensor([1, 2, 3], dtype=torch.long, device=torch.cuda.current_device())

y = l1(x)
z = l2(x)

q = linear(y)
loss = (q + z).sum()

loss.backward()

print("done")

def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.tensor_parallel_word_embeddings:
            input_mask = (input < self.vocab_start_index) | (input >= self.vocab_end_index)
            input = input - self.vocab_start_index
            input[input_mask] = 0

            input = F.embedding(input, self.weight.to_local())

            input[input_mask, :] = 0
            input = tensor_to_dtensor(input, current_placement=Partial())
        else:
            input = F.embedding(input, self.weight.to_local())
            input = tensor_to_dtensor(input, current_placement=Replicate())

        if self.sequence_parallel:
            output_placement = Shard(1)
        else:
            output_placement = Replicate()

        input = dtensor_to_tensor(input, desired_placement=self.output_placement)

        return input