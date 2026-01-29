# torch.rand(B, 64, 2000, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(module, mode="xavier_uniform"):
    if isinstance(module, nn.Conv1d):
        if mode == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

class OnnxDecoderScript(torch.jit.ScriptModule):
    def __init__(self, feat_in, num_classes, init_mode="xavier_uniform"):
        super().__init__()
        self.decoder_layers = nn.Sequential(
            nn.Conv1d(feat_in, num_classes, kernel_size=1, bias=True)
        )
        self.apply(lambda x: init_weights(x, mode=init_mode))  # Initialization preserved

    @torch.jit.script_method
    def forward(self, encoder_output):
        o = self.decoder_layers(encoder_output).transpose(1, 2)
        return F.log_softmax(o, dim=-1)

class MyModel(nn.Module):
    def __init__(self, vocab_size=100):
        super().__init__()
        self.decoder = OnnxDecoderScript(
            feat_in=1024,
            num_classes=vocab_size,
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)[:, :1024]
        x = x.view(-1, 1024, 1)
        return self.decoder(x)

def my_model_function():
    return MyModel()  # Uses default vocab_size=100

def GetInput():
    batch_size = 8  # Matches user's example input dimensions
    return torch.rand(batch_size, 64, 2000, dtype=torch.float32)

