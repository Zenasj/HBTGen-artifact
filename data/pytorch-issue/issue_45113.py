import torch

py
def error(x: Tensor, y: Tensor) -> float:
    assert x.shape == y.shape, f"Different shapes {x.shape} != {y.shape}"
    return (x - y).abs().max().item()

@torch.no_grad()
def test_sspaddmm(batch_size: int, input_size: int, hidden_size: int):
    weight = torch.randn(hidden_size, input_size).to_sparse()
    _bias = torch.randn((hidden_size, 1)).to_sparse()
    bias = torch.cat([_bias] * batch_size, dim=1)
    x = torch.randn(batch_size, input_size)
    y_bis = (bias.to_dense() + weight.to_dense() @ x.t()).to_sparse()
    y = bias.sspaddmm(weight, x.t())

    assert error(y.to_dense(), y_bis.to_dense()) < 1e-6


test_sspaddmm(5, 3, 7)

y_bis = (bias.to_dense() + weight.to_dense() @ x.t()).to_sparse()
y = bias.sspaddmm(weight, x.t())