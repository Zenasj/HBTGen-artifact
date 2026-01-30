import torch
from torch import Tensor
from torch.nn import Module, TransformerEncoderLayer, TransformerEncoder


class TestNet(Module):
    def __init__(self, tf_n_channels=3, device=torch.device('cpu'), *args, **kwargs):
        super().__init__(*args, **kwargs)
        encoder_layer = TransformerEncoderLayer(d_model=tf_n_channels * 5, nhead=tf_n_channels, dim_feedforward=60,
                                                dropout=0.0, device=device, dtype=torch.float32)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, _input: Tensor) -> Tensor:
        return self.transformer(_input, is_causal=True, mask=torch.ones((_input.size(0), _input.size(0)),
                                                                        dtype=torch.bool,
                                                                        device=_input.device).triu(diagonal=1))


if __name__ == '__main__':
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    example_inputs = torch.randn((100, 1, 15)).to(_device, dtype=torch.float)
    model = TestNet(device=_device)
    model = torch.jit.script(model)
    torch.onnx.export(
        model,
        example_inputs,
        "test_model.onnx",
        export_params=True,
        do_constant_folding=True,
        verbose=True
    )