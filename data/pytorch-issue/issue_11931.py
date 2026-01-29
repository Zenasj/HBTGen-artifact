# torch.rand(C, dtype=torch.float32)  # C is the number of categories (e.g., 10,000)
import torch
import torch.nn as nn

class OriginalMultinomial(nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def forward(self, probs):
        return torch.multinomial(probs, self.num_samples, replacement=False)

class AlgorithmAMultinomial(nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def forward(self, probs):
        rand = torch.empty_like(probs).uniform_()
        log_rand_div_p = rand.log() / probs
        _, indices = log_rand_div_p.topk(k=self.num_samples)
        return indices

class MyModel(nn.Module):
    def __init__(self, num_samples=1000):
        super().__init__()
        self.original = OriginalMultinomial(num_samples)
        self.algorithm_a = AlgorithmAMultinomial(num_samples)

    def forward(self, probs):
        sample_orig = self.original(probs)
        sample_alg = self.algorithm_a(probs)
        # Sort indices for comparison (order may differ between methods)
        sorted_orig, _ = torch.sort(sample_orig)
        sorted_alg, _ = torch.sort(sample_alg)
        return torch.all(sorted_orig == sorted_alg)

def my_model_function():
    return MyModel(num_samples=1000)

def GetInput():
    # Generate a 1D tensor of positive weights (probabilities)
    return torch.rand(10000, dtype=torch.float32).clamp_min(0.01)

