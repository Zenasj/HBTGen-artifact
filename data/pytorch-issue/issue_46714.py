import torch.nn as nn
import random

predefined_weights = torch.rand(10, 3)
result = torch.nn.functional.embedding(torch.LongTensor([1,2,0]), predefined_weights, padding_idx=0)

import operator_benchmark as op_bench
import torch
import numpy
from pt import configs

"""EmbeddingBag Operator Benchmark"""

class EmbeddingBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, vocab, dim, input_size, padding, device):
        numpy.random.seed((1 << 32) - 1)
        self.weight = torch.randn(vocab, dim, device=device)
        self.input = torch.tensor(numpy.random.randint(0, vocab, input_size), device=device).long()

        if padding is not None:
            padding_mask = torch.rand(self.input.shape) > 0.5
            self.input[padding_mask] = padding

        self.padding = padding
        self.set_module_name('embedding')

    def forward(self):
        return torch.nn.functional.embedding(self.input, self.weight, padding_idx=self.padding)

op_bench.generate_pt_test(configs.embedding_short_configs, EmbeddingBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()

embedding_short_configs = op_bench.cross_product_configs(
    vocab=[10000, 20000],
    dim=[64, 128],
    padding=[None, 2],
    input_size=[32, 48, 64],
    device=['cpu', 'cuda'],
    tags=['short']
)