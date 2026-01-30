import torch.nn as nn

import torch
from torch.multiprocessing import Process
import copy
def run_model(model, input):
    input_xpu = input.clone().to('xpu')
    model_xpu = copy.deepcopy(model).to('xpu')
    loss_xpu = model_xpu(input_xpu).sum()
    loss = model(input).sum()
    torch.testing.assert_close(loss_xpu.cpu(), loss)
def test_multi_process(model, input):
    p = Process(target=run_model, args=(model, input))
    p.start()
    p.join()
    assert p.exitcode == 0
input = torch.rand(32, 3, 224, 224)
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, 3, stride=2),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
)
if __name__ == "__main__":
    test_multi_process(model, input)
    test_multi_process(model, input)