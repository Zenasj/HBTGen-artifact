import torch
import torch.nn as nn

# torch.rand(5, 10, dtype=torch.float32, device='cuda')  # Input shape (batch_size, 10)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('levels0', torch.tensor([5, 7], device='cuda'))
        self.register_buffer('levels1', torch.tensor([8], device='cuda'))
        self.register_buffer('lit_indices', torch.tensor([0, 1, 2, 3, 4, 6], device='cuda'))
        self.register_buffer('node_indices', torch.tensor([
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[1, 2], [3, 4]],
            [[0, 0], [0, 0]],
            [[1, 4], [9, 9]],
            [[0, 5], [6, 7]]
        ], device='cuda'))
        self.register_buffer('lit_mask0', torch.tensor([0, 1, 2, 1, 2, 0], device='cuda'))
        self.register_buffer('lit_mask1', torch.tensor([1, 1, 0, 0, 1, 0], device='cuda'))

    def forward(self, log_probs):
        lit_weights = torch.stack((log_probs, log_probs), dim=-1).permute(1, 2, 0)
        levels = [self.levels0, self.levels1]
        id_val = 9
        data = torch.zeros(id_val + 1, 5, device='cuda')
        data[id_val] = -1000.0
        data[self.lit_indices] = lit_weights[self.lit_mask0, self.lit_mask1]

        # Process levels[0]
        indices_0 = self.node_indices[levels[0]]
        selected_data = data[indices_0]
        summed_0 = selected_data.sum(dim=-2)
        result_0 = summed_0.logsumexp(dim=-2)
        # Reshape to match levels[0] shape (2,5)
        result_0 = result_0.view(result_0.shape[0], -1).mean(1, keepdim=True).expand(-1, 5)

        data[levels[0]] = result_0

        # Process levels[1]
        indices_1 = self.node_indices[levels[1]]
        selected_data1 = data[indices_1]
        summed1 = selected_data1.sum(dim=-2)
        result_1 = summed1.logsumexp(dim=-2)
        # Reshape to match levels[1] shape (1,5)
        result_1 = result_1.view(1, 5)

        data[levels[1]] = result_1

        return data[levels[-1]]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 10, dtype=torch.float32, device='cuda', requires_grad=True)

