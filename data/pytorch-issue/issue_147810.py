# torch.rand(B, S, dtype=torch.long)  # B=batch size, S=sequence length (e.g., (4,4))
import torch
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.llama.configuration_llama import LlamaConfig

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = LlamaConfig(
            hidden_size=32,
            intermediate_size=88,
            num_hidden_layers=4,
            max_position_embeddings=32,
            vocab_size=512,
            hidden_act='silu',
            rms_norm_eps=1e-6,
        )
        self.model = LlamaModel(config)
        
    def forward(self, input_ids):
        return self.model(input_ids).last_hidden_state

def my_model_function():
    # Returns base model with configuration matching the test case
    return MyModel().cuda()  # Matches original test's .cuda() call

def GetInput():
    # Generates input matching Llama's expected token IDs (integers)
    return torch.randint(low=0, high=512, size=(4,4), dtype=torch.long, device="cuda")

