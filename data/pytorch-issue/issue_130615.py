import torch
import torch.nn as nn


class BasicStatefulModel(nn.Module):
    def __init__(self, dim=64):
        super().__init__()

        self.linear_1 = nn.Linear(dim, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x, state):
        out = state[0]
        if state.shape[1] > 0:
            out = self.linear_1(state[0])
        return self.linear_2(torch.cat([out, x], dim=0))


def test_state_cache_compile():
    dim = 64
    state_size = 2
    window_size = 128
    nstates = 2

    model = torch.compile(BasicStatefulModel(dim).to("cuda"), options={"trace.enabled": True}, disable=False)
    state = torch.randn((nstates, 0, dim), device="cuda")
    for i in range(100):
        x = torch.randn((window_size, dim), device="cuda")
        x = model(x, state)

        if state.shape[1] == 0:
            state = torch.randn((nstates, state_size, dim), device="cuda")
        state[0] = x[-state_size:, :]



if __name__ == "__main__":
    test_state_cache_compile()